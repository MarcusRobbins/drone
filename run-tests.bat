@echo off
call conda activate live-mvp-jax
python -m pip install -U pytest numpy
set PYTHONPATH=%CD%\src
pytest
