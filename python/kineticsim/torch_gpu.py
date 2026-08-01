from __future__ import annotations

import time
from typing import Dict

import torch

from .model import SimConfig, NOISE, MOMENTUM, MAKER, FUNDAMENTAL

class KineticSimTorch:
    name = "torch_gpu"

    def __init__(self, cfg: SimConfig, device: str = "cuda"):
        cfg.validate()
        self.cfg = cfg
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        M, A, L = cfg.n_markets, cfg.n_agents, cfg.n_levels
        dev = self.device
        g = torch.Generator(device=dev)
        g.manual_seed(cfg.seed)
        self.g = g

        self.bid = torch.zeros((M, L), dtype=torch.float32, device=dev)
        self.ask = torch.zeros((M, L), dtype=torch.float32, device=dev)
        self.bid[:, cfg.init_price - 1] = cfg.init_depth
        self.ask[:, cfg.init_price + 1] = cfg.init_depth

        self.last_price = torch.full((M,), float(cfg.init_price), device=dev)
        self.prev_mid = torch.full((M,), float(cfg.init_price), device=dev)

        r = torch.rand((M, A), generator=g, device=dev)
        atype = torch.empty((M, A), dtype=torch.int32, device=dev)
        atype[:] = NOISE
        atype[r >= cfg.frac_noise] = MOMENTUM
        atype[r >= (cfg.frac_noise + cfg.frac_momentum)] = MAKER
        atype[r >= (cfg.frac_noise + cfg.frac_momentum + cfg.frac_maker)] = FUNDAMENTAL
        self.agent_type = atype

        self.arange_L = torch.arange(L, device=dev)
        self.arange_A = torch.arange(A, device=dev)
        self.step_idx = 0
        self.total_volume = torch.zeros(M, dtype=torch.float64, device=dev)
        self.n_trades = torch.zeros(M, dtype=torch.int64, device=dev)

    def _best_bid_ask(self):
        L = self.cfg.n_levels
        has_b = (self.bid > 0).any(dim=1)
        has_a = (self.ask > 0).any(dim=1)
        best_bid = (L - 1) - torch.argmax((self.bid > 0).flip(1).int(), dim=1)
        best_ask = torch.argmax((self.ask > 0).int(), dim=1)
        return best_bid, best_ask, has_b, has_a

    @torch.no_grad()
    def step(self) -> None:
        cfg = self.cfg
        M, A, L = cfg.n_markets, cfg.n_agents, cfg.n_levels
        dev = self.device
        g = self.g

        bb, ba, has_b, has_a = self._best_bid_ask()
        mid = torch.where(has_b & has_a, 0.5 * (bb + ba).float(), self.last_price)
        ret = torch.sign(mid - self.prev_mid)
        self.prev_mid = mid.clone()

        midA = mid[:, None]
        retA = ret[:, None]
        typ = self.agent_type

        u_side = torch.rand((M, A), generator=g, device=dev)
        u_mkt = torch.rand((M, A), generator=g, device=dev)
        qty = torch.randint(1, int(cfg.max_order_qty) + 1, (M, A),
                            generator=g, device=dev).float()
        noise_off = torch.randint(-cfg.noise_spread, cfg.noise_spread + 1, (M, A),
                                  generator=g, device=dev)

        side = torch.where(u_side < 0.5, 1, -1).int()
        price = torch.round(midA + noise_off).int()

        is_mom = typ == MOMENTUM
        mom_side = torch.where(retA != 0, retA.int(), side)
        side = torch.where(is_mom, mom_side, side)
        price = torch.where(is_mom, torch.round(midA + mom_side.float()).int(), price)

        is_fund = typ == FUNDAMENTAL
        dev_f = float(cfg.init_price) - midA
        fund_side = torch.where(dev_f > 0, 1, torch.where(dev_f < 0, -1, side)).int()
        side = torch.where(is_fund, fund_side, side)
        price = torch.where(is_fund, torch.round(midA + fund_side.float()).int(), price)

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
        self.bid = BUY - traded_buy
        self.ask = SELL - traded_sell

        traded = volume > 0
        self.last_price = torch.where(traded, pstar.float(), self.last_price)
        self.total_volume += volume.double()
        self.n_trades += traded.long()
        self.step_idx += 1

    @torch.no_grad()
    def run(self, n_steps: int | None = None) -> Dict:
        n = n_steps if n_steps is not None else self.cfg.n_steps
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n):
                self.step()
            end.record()
            torch.cuda.synchronize()
            dt = start.elapsed_time(end) / 1000.0
        else:
            t0 = time.perf_counter()
            for _ in range(n):
                self.step()
            dt = time.perf_counter() - t0
        return self.stats(n, dt)

    def stats(self, n_steps: int, wall_time: float) -> Dict:
        cfg = self.cfg
        events = cfg.n_markets * cfg.n_agents * n_steps
        mem = (torch.cuda.max_memory_allocated() / 1e9
               if self.device.type == "cuda" else 0.0)
        return {
            "backend": self.name,
            "n_markets": cfg.n_markets,
            "n_agents": cfg.n_agents,
            "n_levels": cfg.n_levels,
            "n_steps": n_steps,
            "wall_time_s": wall_time,
            "events": events,
            "events_per_s": events / wall_time if wall_time > 0 else float("nan"),
            "steps_per_s": n_steps / wall_time if wall_time > 0 else float("nan"),
            "gpu_mem_gb": mem,
            "mean_last_price": float(self.last_price.mean().item()),
            "std_last_price": float(self.last_price.std().item()),
            "mean_volume_per_market": float(self.total_volume.mean().item()),
            "mean_trades_per_market": float(self.n_trades.double().mean().item()),
        }

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

        is_fund = typ == FUNDAMENTAL
        dev_f = float(cfg.init_price) - midA
        fund_side = torch.where(dev_f > 0, 1, torch.where(dev_f < 0, -1, side)).int()
        side = torch.where(is_fund, fund_side, side)
        price = torch.where(is_fund, torch.round(midA + fund_side.float()).int(), price)

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
        self.captured_steps = 0

    @torch.no_grad()
    def _reset_in_place(self) -> None:
        cfg = self.cfg
        self.bid.zero_()
        self.ask.zero_()
        self.bid[:, cfg.init_price - 1] = cfg.init_depth
        self.ask[:, cfg.init_price + 1] = cfg.init_depth
        self.last_price.fill_(float(cfg.init_price))
        self.prev_mid.fill_(float(cfg.init_price))
        self.total_volume.zero_()
        self.n_trades.zero_()
        self.step_idx = 0

    @torch.no_grad()
    def run(self, n_steps: int | None = None) -> Dict:
        n = n_steps if n_steps is not None else self.cfg.n_steps

        if self.g_captured is None or self.captured_steps != n:
            self._reset_in_place()
            for _ in range(5):
                self.step()

            self._reset_in_place()
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

        self._reset_in_place()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.g_captured.replay()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return self.stats(n, dt)

class KineticSimTorchCompile(KineticSimTorchInPlace):
    name = "torch_compile"

    def __init__(self, cfg: SimConfig, device: str = "cuda"):
        super().__init__(cfg, device)
        self.step = torch.compile(self.step, backend="cudagraphs")
