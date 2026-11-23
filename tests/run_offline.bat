@echo off
rem The tests that need nothing running (no model, no app), on Windows: tests\run_offline.bat
rem (on Linux or macOS: sh tests/run_offline.sh). Exits with the number of test files that failed.
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf8
if "%PYTHON%"=="" (where py >nul 2>nul && (set PYTHON=py) || (set PYTHON=python))
set failed=0
for %%t in (test_rich_points test_together test_stuck test_sizing_notes test_hub_split test_fetch test_kiwix_names) do (
    %PYTHON% tests\%%t.py > "%TEMP%\rapidli_%%t.txt" 2>&1
    if errorlevel 1 (
        echo FAIL  %%t
        findstr /b "FAIL" "%TEMP%\rapidli_%%t.txt"
        set /a failed+=1
    ) else (
        echo ok    %%t
    )
    del "%TEMP%\rapidli_%%t.txt" >nul 2>nul
)
echo !failed! test file(s) failed
exit /b !failed!
