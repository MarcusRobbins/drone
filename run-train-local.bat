@echo off
setlocal
REM Run without installing package: add src to PYTHONPATH
set PYTHONPATH=%CD%\src
python -m live_mvp.train_live
