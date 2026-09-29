# Launcher for the Homework-3 QA page (Flask, fully offline).
#
# IMPORTANT: this file is intentionally 100% ASCII (comments included).
# Windows PowerShell 5.1 decodes UTF-8-without-BOM scripts as GBK on Chinese
# Windows, so any non-ASCII character (for example a Chinese file name) can be
# mangled and fail with "The string is missing the terminator" - which is why the
# first version of this launcher could not start the page at all.
# The page script is therefore located by its Flask route marker instead of by name.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\start_qa_page.ps1
#   or double-click the .cmd launcher in the same folder
#   then open the URL printed at the end (default http://127.0.0.1:8000)
#
# Optional environment variables:
#   QA_PYTHON          python.exe path (default E:\Anaconda\python.exe)
#   QA_PORT            port (default 8000; auto-increments if busy)
#   MODEL_DIR          override the embedding model directory
#   DEEPSEEK_API_KEY   optional generative answering (off by default)

$ErrorActionPreference = "Stop"
# Make the console print Chinese file names correctly (cosmetic only)
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$codeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if ($env:QA_PYTHON) { $env:QA_PYTHON } else { "E:\Anaconda\python.exe" }
$port = if ($env:QA_PORT) { [int]$env:QA_PORT } else { 8000 }

if (-not (Test-Path $python)) {
    throw "Python not found: $python (set QA_PYTHON to your python.exe)"
}

# Locate the page script by content, never by a non-ASCII file name
$page = Get-ChildItem -LiteralPath $codeDir -Filter *.py | Where-Object {
    (Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8) -match '@app\.post\("/api/ask"\)'
} | Select-Object -First 1

if (-not $page) {
    throw "QA page script not found in $codeDir (expected a .py containing the Flask /api/ask route)"
}

Write-Host "Starting QA page: $($page.Name)" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor DarkGray
Write-Host "The first run needs to load the embedding model (offline, ~10s); the page will tell you when it is ready." -ForegroundColor DarkGray

& $python $page.FullName --port $port --open
