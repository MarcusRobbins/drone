from __future__ import annotations
import os, textwrap


def _default_cache_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser(r"~\AppData\Local"))
        return os.path.join(base, "jax")
    else:
        base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        return os.path.join(base, "jax")


def enable_persistent_cache(cache_dir: str | None = None) -> str:
    """
    Initialize JAX's persistent compilation cache.
    Tries public API first, then private fallbacks. No-ops if unsupported.
    Returns a status string (never raises).
    """
    path = cache_dir or os.environ.get("JAX_CACHE_DIR") or _default_cache_dir()
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

    tried: list[str] = []
    # 1) Public API (newer JAX)
    try:
        from jax.experimental import compilation_cache as cc  # type: ignore[attr-defined]
        for name in ("initialize_cache", "initialize_cache_if_needed", "set_cache_dir"):
            fn = getattr(cc, name, None)
            if callable(fn):
                fn(path)
                return f"persistent cache: {cc.__name__}.{name} at {path}"
            tried.append(f"jax.experimental.compilation_cache.{name}=MISSING")
    except Exception as e:
        tried.append(f"import jax.experimental.compilation_cache failed: {type(e).__name__}: {e}")

    # 2) Private API (older JAX, use best-effort)
    try:
        from jax._src import compilation_cache as cc  # type: ignore
        for name in ("initialize_cache", "initialize_cache_if_needed", "set_cache_dir"):
            fn = getattr(cc, name, None)
            if callable(fn):
                fn(path)
                return f"persistent cache: {cc.__name__}.{name} at {path}"
            tried.append(f"jax._src.compilation_cache.{name}=MISSING")
    except Exception as e:
        tried.append(f"import jax._src.compilation_cache failed: {type(e).__name__}: {e}")

    return "persistent cache unavailable; " + "; ".join(tried)
