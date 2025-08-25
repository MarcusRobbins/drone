from __future__ import annotations
import time
from typing import Any, Dict, List

import jax


def _block(x: Any) -> Any:
    """Synchronize with the device for accurate timing (handles pytrees)."""
    try:
        jax.block_until_ready(x)
        return x
    except Exception:
        try:
            for leaf in jax.tree_util.tree_leaves(x):
                try:
                    jax.block_until_ready(leaf)
                except Exception:
                    pass
        except Exception:
            pass
        return x


def time_blocked(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), block until device work is done, return (out, dt_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _block(out)
    return out, (time.perf_counter() - t0)


class StepTimes:
    def __init__(self, name: str = "train_step"):
        self.name = name
        self._ts: List[float] = []

    def add(self, dt: float) -> None:
        self._ts.append(float(dt))

    def summary(self) -> Dict[str, float]:
        if not self._ts:
            return {}
        xs = sorted(self._ts)
        n = len(xs)

        def pct(p: float) -> float:
            if n == 1:
                return xs[0]
            idx = min(max(int(p * (n - 1)), 0), n - 1)
            return xs[idx]

        return {
            "count": float(n),
            "min": xs[0],
            "mean": sum(xs) / n,
            "median": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "max": xs[-1],
        }

