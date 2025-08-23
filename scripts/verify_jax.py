
import jax, jax.numpy as jnp

print("JAX version:", jax.__version__)
try:
    import jaxlib
    print("jaxlib version:", jaxlib.__version__)
except Exception as e:
    print("jaxlib not importable:", e)

x = jnp.arange(9, dtype=jnp.float32).reshape(3,3)
y = x @ x.T
print("devices:", jax.devices())
print("platform:", jax.default_backend())
print("matmul result (sum):", float(y.sum()))

# Simple GPU check
gpu = any(d.platform == "gpu" for d in jax.devices())
print("GPU available:", gpu)
