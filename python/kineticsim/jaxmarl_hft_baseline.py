from __future__ import annotations

import os
import sys
import time
from typing import Dict

import numpy as np

from .model import SimConfig

_JAXMARL_COMPILE_CACHE = {}

class JaxMarlHFTBaseline:
    name = "jaxmarl_hft"

    def __init__(self, cfg: SimConfig, env_id: str | None = None):
        cfg.validate()
        self.cfg = cfg

    def _build(self):
        import jax
        import jax.numpy as jnp
        from jax import random as jrandom

        cfg = self.cfg
        cache_key = (cfg.n_agents, cfg.n_steps, cfg.seed)
        global _JAXMARL_COMPILE_CACHE
        if cache_key in _JAXMARL_COMPILE_CACHE:
            batched, A_env = _JAXMARL_COMPILE_CACHE[cache_key]
            return jax, jrandom, batched, A_env

        import sys
        for k in list(sys.modules.keys()):
            if k == "gymnax_exchange" or k.startswith("gymnax_exchange."):
                del sys.modules[k]
        import platform
        is_windows = platform.system() == "Windows"
        base_path = "E:/Agent/baselines" if is_windows else "/mnt/e/Agent/baselines"

        path_to_add = f"{base_path}/JaxMARL-HFT"
        if path_to_add not in sys.path:
            sys.path.insert(0, path_to_add)
        for p in list(sys.path):
            if "jax-lob" in p or "jaxlob" in p.lower():
                try:
                    sys.path.remove(p)
                except ValueError:
                    pass

        try:
            from gymnax_exchange.jaxen.marl_env import MARLEnv
            from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig, World_EnvironmentConfig, MarketMaking_EnvironmentConfig, Execution_EnvironmentConfig
        except ImportError as e:
            raise ImportError(
                "Could not construct the JaxMARL-HFT environment. Clone "
                f"https://github.com/vmohl/JaxMARL-HFT and configure path. "
                f"Underlying error: {e}"
            )

        M, S = cfg.n_markets, cfg.n_steps

        world_config = World_EnvironmentConfig(
            stock="AMZN",
            timePeriod="2017Jan_oneday",
            dataPath=f"{base_path}/JaxMARL-HFT/data",
            alphatradePath=f"{base_path}/JaxMARL-HFT",
            seed=cfg.seed,
            episode_time=cfg.n_steps,
            ep_type="fixed_steps"
        )

        n_mm = max(1, cfg.n_agents // 2)
        n_exe = max(1, cfg.n_agents - n_mm)

        ma_config = MultiAgentConfig(
            number_of_agents_per_type=[n_mm, n_exe],
            dict_of_agents_configs={
                "MarketMaking": MarketMaking_EnvironmentConfig(normalize=True),
                "Execution": Execution_EnvironmentConfig(normalize=True)
            },
            world_config=world_config
        )

        env = MARLEnv(key=jrandom.PRNGKey(cfg.seed), multi_agent_config=ma_config)
        params = env.default_params

        A_env = sum(env.multi_agent_config.number_of_agents_per_type)

        def sample_actions(key):
            subkeys = jrandom.split(key, len(env.action_spaces))
            actions = [
                jax.vmap(space.sample)(
                    jax.random.split(sk, n_agents)
                )
                for sk, space, n_agents in zip(
                    subkeys,
                    env.action_spaces,
                    env.multi_agent_config.number_of_agents_per_type,
                )
            ]
            return actions

        def one_env_rollout(key):
            key, rk = jrandom.split(key)
            obs, state = env.reset(rk, params)

            def step_fn(carry, _):
                st, k = carry
                k, ak, sk = jrandom.split(k, 3)
                actions = sample_actions(ak)
                obs, st, rew, done, info = env.step(sk, st, actions, params)
                return (st, k), None

            (state, _), _ = jax.lax.scan(step_fn, (state, key), None, length=S)
            return state

        batched = jax.jit(jax.vmap(one_env_rollout))
        _JAXMARL_COMPILE_CACHE[cache_key] = (batched, A_env)
        return jax, jrandom, batched, A_env

    def run(self, n_steps: int | None = None) -> Dict:
        cfg = self.cfg
        if n_steps is not None and n_steps != cfg.n_steps:
            from dataclasses import replace
            self.cfg = replace(cfg, n_steps=n_steps)
            cfg = self.cfg
        jax, jrandom, batched, A_env = self._build()

        M_run = min(cfg.n_markets, 8)
        keys = jrandom.split(jrandom.PRNGKey(cfg.seed), M_run)

        out = batched(keys); jax.block_until_ready(out)
        t0 = time.perf_counter()
        out = batched(keys); jax.block_until_ready(out)
        dt_run = time.perf_counter() - t0

        dt = dt_run * (cfg.n_markets / M_run)

        events = cfg.n_markets * A_env * cfg.n_steps
        nan = float("nan")
        return {
            "backend": self.name,
            "n_markets": cfg.n_markets, "n_agents": A_env,
            "n_levels": cfg.n_levels, "n_steps": cfg.n_steps,
            "wall_time_s": dt, "events": events,
            "events_per_s": events / dt if dt > 0 else nan,
            "steps_per_s": cfg.n_steps / dt if dt > 0 else nan,
            "us_per_step": 1e6 * dt / cfg.n_steps,
            "ns_per_event": 1e9 * dt / events if events > 0 else nan,
            "gpu_mem_gb": nan,
            "mean_last_price": nan, "std_last_price": nan,
            "mean_volume_per_market": nan, "mean_trades_per_market": nan,
        }

if __name__ == "__main__":
    cfg = SimConfig(n_markets=16, n_agents=16, n_steps=10)
    s = JaxMarlHFTBaseline(cfg).run()
    print(f"JaxMARL-HFT: {s['events_per_s']:.3e} agent-steps/s, "
          f"{s['us_per_step']:.1f} us/step")
