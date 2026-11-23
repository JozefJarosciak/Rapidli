@echo off
rem Starts Rapidli (Rapid Local Inference) at http://127.0.0.1:11842 and opens it in the browser.
rem Rapidli runs in the background (no window); its log is data\logs\rapidli.log. It loads the model used last
rem (Settings > Engine > "Load the last model at start") with as many lanes as the graphics card holds.
rem stop.bat stops Rapidli and its engine.
setlocal
cd /d "%~dp0"
rem Python: the py launcher when it is installed, else python on the PATH (set RAPIDLI_PYTHON to choose one)
set PY=py
where py >nul 2>nul || set PY=python
if defined RAPIDLI_PYTHON set PY=%RAPIDLI_PYTHON%
"%PY%" --version >nul 2>nul
if errorlevel 1 (
  echo Python was not found: install Python 3.10 or newer from https://www.python.org/downloads/ ^(keep "py launcher" ticked^),
  echo or set RAPIDLI_PYTHON to the python.exe to use.
  exit /b 1
)
"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Rapidli needs Python 3.10 or newer: install it from https://www.python.org/downloads/ or set RAPIDLI_PYTHON.
  exit /b 1
)
if not exist engine\bin\rapidli-engine.exe (
  echo The engine is not installed: download rapidli-engine-windows-x64.zip from the Releases page and unzip it
  echo into engine\bin, so that engine\bin\rapidli-engine.exe exists ^(see README.md^).
  exit /b 1
)
powershell -NoProfile -Command "try { Invoke-RestMethod -TimeoutSec 2 http://127.0.0.1:11842/api/status | Out-Null; exit 1 } catch { exit 0 }"
if errorlevel 1 (
  echo Rapidli is already running: opening it.
  start "" http://127.0.0.1:11842
  exit /b 0
)
if not exist data\logs mkdir data\logs
powershell -NoProfile -Command "Start-Process -FilePath '%PY%' -ArgumentList '-X','utf8','app\rapidli.py','--start','--open' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput 'data\logs\rapidli.log' -RedirectStandardError 'data\logs\rapidli.err.log'"
echo Rapidli is starting at http://127.0.0.1:11842 (the model takes 20 to 60 seconds to load).
