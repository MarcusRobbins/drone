def test_imports():
    import jax
    import live_mvp
    assert hasattr(jax, "__version__")
