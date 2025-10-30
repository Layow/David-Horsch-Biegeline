@echo off
setlocal enableextensions enabledelayedexpansion

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo Virtual environment not found. Please run first-run.bat first.
  exit /b 1
)

"%VENV_PY%" symbolic_main_parallel.py %*

endlocal

