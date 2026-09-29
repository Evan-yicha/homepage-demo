# ASCII-named launcher for the QA page (same purpose as the Chinese-named .ps1).
#
# Why two launchers: Windows PowerShell 5.1 decodes scripts that lack a BOM as
# GBK on Chinese Windows. A script that contains a Chinese *file name* would then
# be mangled and fail with "The string is missing the terminator". Both launcher
# files are therefore pure ASCII and locate the page script by its Flask route.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\start_qa_page.ps1
#   (or double-click the .cmd launcher in the same folder)

$ErrorActionPreference = "Stop"
# Make the console print Chinese file names correctly (cosmetic only)
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$codeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if ($env:QA_PYTHON) { $env:QA_PYTHON } else { "E:\Anaconda\python.exe" }
$port = if ($env:QA_PORT) { [int]$env:QA_PORT } else { 8000 }

if (-not (Test-Path $python)) {
    throw "Python not found: $python (set QA_PYTHON to your python.exe)"
}

$page = Get-ChildItem -LiteralPath $codeDir -Filter *.py | Where-Object {
    (Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8) -match '@app\.post\("/api/ask"\)'
} | Select-Object -First 1

if (-not $page) {
    throw "QA page script not found in $codeDir (expected a .py containing the Flask /api/ask route)"
}

Write-Host "Starting QA page: $($page.Name)" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor DarkGray
Write-Host "Loading the local embedding model takes ~10s on first start; the page shows a 'loading' badge until it is ready." -ForegroundColor DarkGray

& $python $page.FullName --port $port --open
