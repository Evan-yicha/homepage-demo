@echo off
rem ============================================================
rem  Homework-3 QA page launcher (double-click to run).
rem  ASCII-only on purpose: cmd/PowerShell on Chinese Windows may
rem  mis-decode UTF-8 without BOM, which is exactly why the old
rem  .ps1 failed with "missing the terminator".
rem ============================================================
chcp 65001 >nul
setlocal
set "CODEDIR=%~dp0"
if "%QA_PYTHON%"=="" set "QA_PYTHON=E:\Anaconda\python.exe"

if not exist "%QA_PYTHON%" (
  echo [ERROR] Python not found: %QA_PYTHON%
  echo         Set QA_PYTHON to your python.exe path and retry.
  pause
  exit /b 1
)

echo Starting QA page with %QA_PYTHON% ...
echo The first question may take a few seconds while the local embedding model loads.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%CODEDIR%start_qa_page.ps1"

echo.
echo QA page stopped.
pause
