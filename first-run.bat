@echo off
setlocal enableextensions enabledelayedexpansion

rem Determine a Python launcher
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PY=py -3"
 ) else (
  set "PY=python"
)

%PY% -m venv .venv

set "VENV_PY=.venv\Scripts\python.exe"

"%VENV_PY%" -m pip install -U pip
"%VENV_PY%" -m pip install -r requirements.txt

echo Venv ready. Activate with: .\.venv\Scripts\activate
echo Run scripts manually, e.g.: python numeric_main.py

endlocal
