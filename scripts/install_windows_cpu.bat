@echo off
setlocal enabledelayedexpansion

echo === live_mvp :: Windows (CPU-only) install ===

REM --- Ensure we're in the project root (must contain pyproject.toml)
if not exist "pyproject.toml" (
  echo [ERROR] pyproject.toml not found. Run this script from the project root.
  exit /b 1
)

REM --- Check that Conda is available
where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Conda not found. Please open an "Anaconda Prompt" or install Miniconda/Anaconda.
  exit /b 1
)

REM --- Remove any existing env (ignore errors if it doesn't exist)
echo Removing existing env 'live-mvp-jax' if present...
call conda env remove -y -n live-mvp-jax >nul 2>&1

REM --- Create fresh env
echo Creating env 'live-mvp-jax'...
call conda create -y -n live-mvp-jax python=3.11 pip
if errorlevel 1 (
  echo [ERROR] Failed to create conda env.
  exit /b 1
)

REM --- Activate env
call conda activate live-mvp-jax
if errorlevel 1 (
  echo [ERROR] Failed to activate conda env 'live-mvp-jax'.
  exit /b 1
)

REM --- Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip/setuptools/wheel.
  exit /b 1
)

REM --- Install JAX (CPU) + Optax
REM On Windows, GPU wheels aren't supported; this installs CPU-only jax/jaxlib.
python -m pip install -U jax jaxlib optax
if errorlevel 1 (
  echo [WARN] 'pip install jax jaxlib optax' failed. Trying fallback 'pip install jax optax'...
  python -m pip install -U jax optax
  if errorlevel 1 (
    echo [ERROR] Failed to install JAX. See https://github.com/google/jax#installation
    exit /b 1
  )
)

REM --- Editable install of this package
python -m pip install -e .
if errorlevel 1 (
  echo [ERROR] Editable install failed. Ensure 'pyproject.toml' is in this folder.
  exit /b 1
)

REM --- Verify JAX (and that the env has the package)
if exist "verify_jax.py" (
  echo.
  echo === Verifying JAX devices ===
  python verify_jax.py
) else (
  echo.
  echo [INFO] 'verify_jax.py' not found; quick inline check:
  python - <<PY
import jax, importlib, sys
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
m = importlib.import_module("live_mvp")
print("Imported live_mvp from:", m.__file__)
PY
)

echo.
echo Done. In VS Code, select the interpreter named: live-mvp-jax