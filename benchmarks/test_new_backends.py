import sys
import os
import time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from kineticsim.model import SimConfig, NOISE, MOMENTUM, MAKER
from kineticsim.torch_gpu import KineticSimTorch

class KineticSimTorchInPlace(KineticSimTorch):
    name = "torch_inplace"

    def reset(self) -> None:
        super().reset()
        torch.cuda.manual_seed(self.cfg.seed)

    @torch.no_grad()
    def step(self) -> None:
        cfg = self.cfg
        M, A, L = cfg.n_markets, cfg.n_agents, cfg.n_levels
        dev = self.device

        bb, ba, has_b, has_a = self._best_bid_ask()
        mid = torch.where(has_b & has_a, 0.5 * (bb + ba).float(), self.last_price)
        ret = torch.sign(mid - self.prev_mid)
        self.prev_mid.copy_(mid)

        midA = mid[:, None]
        retA = ret[:, None]
        typ = self.agent_type

        u_side = torch.rand((M, A), device=dev)
        u_mkt = torch.rand((M, A), device=dev)
        qty = torch.randint(1, int(cfg.max_order_qty) + 1, (M, A), device=dev).float()
        noise_off = torch.randint(-cfg.noise_spread, cfg.noise_spread + 1, (M, A), device=dev)

        side = torch.where(u_side < 0.5, 1, -1).int()
        price = torch.round(midA + noise_off).int()

        is_mom = typ == MOMENTUM
        mom_side = torch.where(retA != 0, retA.int(), side)
        side = torch.where(is_mom, mom_side, side)
        price = torch.where(is_mom, torch.round(midA + mom_side.float()).int(), price)

        is_mm = typ == MAKER
        mm_buy = ((self.arange_A[None, :] + self.step_idx) % 2) == 0
        mm_side = torch.where(mm_buy, 1, -1).int()
        mm_price = torch.round(
            midA + torch.where(mm_buy, -float(cfg.maker_half_spread),
                               float(cfg.maker_half_spread))
        ).int()
        side = torch.where(is_mm, mm_side, side)
        price = torch.where(is_mm, mm_price, price)

        is_market = (u_mkt < cfg.market_order_prob) & (~is_mm)
        price = torch.where(is_market & (side > 0), L - 1, price)
        price = torch.where(is_market & (side < 0), 0, price)
        price = price.clamp(0, L - 1).long()

        incoming_buy = torch.zeros((M, L), dtype=torch.float32, device=dev)
        incoming_sell = torch.zeros((M, L), dtype=torch.float32, device=dev)
        buy_mask = (side > 0).float()
        sell_mask = (side <= 0).float()
        incoming_buy.scatter_add_(1, price, qty * buy_mask)
        incoming_sell.scatter_add_(1, price, qty * sell_mask)

        BUY = self.bid + incoming_buy
        SELL = self.ask + incoming_sell

        Dcum = torch.flip(torch.cumsum(torch.flip(BUY, [1]), dim=1), [1])
        Scum = torch.cumsum(SELL, dim=1)
        match = torch.minimum(Dcum, Scum)
        volume, pstar = match.max(dim=1)
        V = volume[:, None]

        traded_buy = torch.clamp(V - (Dcum - BUY), min=0.0).minimum(BUY)
        traded_sell = torch.clamp(V - (Scum - SELL), min=0.0).minimum(SELL)

        self.bid.copy_(BUY - traded_buy)
        self.ask.copy_(SELL - traded_sell)

        traded = volume > 0
        self.last_price.copy_(torch.where(traded, pstar.float(), self.last_price))
        self.total_volume.add_(volume.double())
        self.n_trades.add_(traded.long())
        self.step_idx += 1

class KineticSimTorchGraph(KineticSimTorchInPlace):
    name = "torch_graph"

    def __init__(self, cfg: SimConfig, device: str = "cuda"):
        super().__init__(cfg, device)
        self.g_captured = None

    def run(self, n_steps: int | None = None) -> dict:
        n = n_steps if n_steps is not None else self.cfg.n_steps

        if self.g_captured is None or self.captured_steps != n:
            self.reset()
            for _ in range(5):
                self.step()

            self.reset()
            torch.cuda.synchronize()
            self.g_captured = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.g_captured.capture_begin()
                for _ in range(n):
                    self.step()
                self.g_captured.capture_end()
            torch.cuda.current_stream().wait_stream(s)
            self.captured_steps = n
            torch.cuda.synchronize()

        self.reset()
        t0 = time.perf_counter()
        self.g_captured.replay()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return self.stats(n, dt)

def test_all():
    cfg = SimConfig(n_markets=8192, n_agents=256, n_steps=500)

    print("Running baseline PyTorch...")
    sim_native = KineticSimTorch(cfg)
    sim_native.run(50)
    res_native = sim_native.run()
    print(f"Native PyTorch: {res_native['wall_time_s']*1000:.2f} ms ({res_native['events_per_s']:.3e} ev/s)")

    print("Running in-place PyTorch...")
    sim_inplace = KineticSimTorchInPlace(cfg)
    sim_inplace.run(50)
    res_inplace = sim_inplace.run()
    print(f"In-place PyTorch: {res_inplace['wall_time_s']*1000:.2f} ms ({res_inplace['events_per_s']:.3e} ev/s)")

    print("Running CUDA Graph PyTorch...")
    sim_graph = KineticSimTorchGraph(cfg)
    sim_graph.run()
    res_graph = sim_graph.run()
    print(f"CUDA Graph PyTorch: {res_graph['wall_time_s']*1000:.2f} ms ({res_graph['events_per_s']:.3e} ev/s)")

    print("Running torch.compile (cudagraphs backend) PyTorch...")
    sim_compile = KineticSimTorchInPlace(cfg)
    sim_compile.step = torch.compile(sim_compile.step, backend="cudagraphs")
    sim_compile.run(50)
    res_compile = sim_compile.run()
    print(f"torch.compile (cudagraphs): {res_compile['wall_time_s']*1000:.2f} ms ({res_compile['events_per_s']:.3e} ev/s)")

if __name__ == "__main__":
    test_all()
