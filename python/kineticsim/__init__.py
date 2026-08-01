from .model import SimConfig, NOISE, MOMENTUM, MAKER, SWEEP_MARKETS, SWEEP_AGENTS
from .reference_cpu import KineticSimCPU, call_auction

__all__ = [
    "SimConfig", "NOISE", "MOMENTUM", "MAKER",
    "SWEEP_MARKETS", "SWEEP_AGENTS",
    "KineticSimCPU", "call_auction",
    "get_backend",
]

__version__ = "0.1.0"

def get_backend(name: str):
    name = name.lower()
    if name in ("cpu", "cpu_numpy", "numpy"):
        return KineticSimCPU
    if name in ("torch", "torch_gpu", "pytorch"):
        from .torch_gpu import KineticSimTorch
        return KineticSimTorch
    if name in ("jax", "jax_gpu"):
        from .jax_gpu import KineticSimJax
        return KineticSimJax
    if name in ("cuda", "kineticsim", "optimized"):
        from .cuda_backend import KineticSimCUDA
        return KineticSimCUDA
    if name in ("naive", "naive_cuda"):
        from .cuda_backend import KineticSimNaiveCUDA
        return KineticSimNaiveCUDA
    raise ValueError(f"unknown backend: {name!r}")
