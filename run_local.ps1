# Kage - From Ink to Imagination. Starts backend and frontend in two visible terminals.
#
#   .\run_local.ps1            start both, open the browser
#   .\run_local.ps1 -NoBrowser start both, do not open the browser
#   .\run_local.ps1 -Stop      stop anything already listening on 8000 / 8501

param(
    [switch]$NoBrowser,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backendPort = 8000
$frontendPort = 8501

function Stop-Port([int]$Port, [string]$Label) {
    # NOTE: $pid is a read-only automatic variable in PowerShell - do not reuse that name.
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($procId in ($conns | Select-Object -ExpandProperty OwningProcess -Unique)) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "  stopped $Label (PID $procId) on port $Port" -ForegroundColor DarkYellow
        } catch {
            Write-Host "  could not stop PID $procId on port $Port : $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

if ($Stop) {
    Write-Host "Stopping Kage services..." -ForegroundColor Cyan
    Stop-Port $backendPort  "backend"
    Stop-Port $frontendPort "frontend"
    Write-Host "Done." -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------- interpreter

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No virtual environment at .venv" -ForegroundColor Yellow
    Write-Host "Create one with:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

# ------------------------------------------------------------- pre-flight

Write-Host "Checking environment..." -ForegroundColor Cyan
& $python (Join-Path $root "scripts\check_environment.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nEnvironment check failed. Fix the items above before starting." -ForegroundColor Red
    exit 1
}

Write-Host "`nFreeing ports if they are already in use..." -ForegroundColor Cyan
Stop-Port $backendPort  "backend"
Stop-Port $frontendPort "frontend"

New-Item -ItemType Directory -Force (Join-Path $root "data\logs") | Out-Null

# --------------------------------------------------------------- launch

# Each service gets its own PowerShell window that stays open (-NoExit) so errors
# remain on screen instead of vanishing with the process.

Write-Host "Opening backend terminal..." -ForegroundColor Cyan
$backendCmd = @"
`$Host.UI.RawUI.WindowTitle = 'KAGE - BACKEND (port $backendPort)'
Set-Location '$root'
Write-Host '=== KAGE :: FastAPI backend ===' -ForegroundColor Magenta
Write-Host 'API  : http://127.0.0.1:$backendPort'
Write-Host 'Docs : http://127.0.0.1:$backendPort/docs'
Write-Host 'Logs : $root\data\logs\kage.log'
Write-Host 'Press Ctrl+C to stop.' -ForegroundColor DarkGray
Write-Host ''
& '$python' '$root\main.py'
Write-Host ''
Write-Host 'Backend stopped. This window stays open so you can read any error above.' -ForegroundColor Yellow
"@
Start-Process powershell -ArgumentList "-NoExit", "-NoProfile", "-Command", $backendCmd

Write-Host "Waiting for the backend to answer..." -ForegroundColor Cyan
$backendUp = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 750
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$backendPort/api/health" -TimeoutSec 3
        Write-Host "  backend is up (status: $($r.status), render_ready: $($r.render_ready))" -ForegroundColor Green
        $backendUp = $true
        break
    } catch { }
}
if (-not $backendUp) {
    Write-Host "  backend did not come up in 30s - check its window and data\logs\errors.log" -ForegroundColor Red
}

Write-Host "Opening frontend terminal..." -ForegroundColor Cyan
$streamlit = Join-Path $root ".venv\Scripts\streamlit.exe"
$frontendLaunch = if (Test-Path $streamlit) {
    "& '$streamlit' run '$root\app.py' --server.port $frontendPort --browser.gatherUsageStats false"
} else {
    "& '$python' -m streamlit run '$root\app.py' --server.port $frontendPort --browser.gatherUsageStats false"
}
$frontendCmd = @"
`$Host.UI.RawUI.WindowTitle = 'KAGE - FRONTEND (port $frontendPort)'
Set-Location '$root'
Write-Host '=== KAGE :: Streamlit frontend ===' -ForegroundColor Magenta
Write-Host 'App : http://localhost:$frontendPort'
Write-Host 'Press Ctrl+C to stop.' -ForegroundColor DarkGray
Write-Host ''
$frontendLaunch
Write-Host ''
Write-Host 'Frontend stopped. This window stays open so you can read any error above.' -ForegroundColor Yellow
"@
Start-Process powershell -ArgumentList "-NoExit", "-NoProfile", "-Command", $frontendCmd

# Streamlit's first start compiles a lot of imports; 90s is a realistic cold-start budget.
Write-Host "Waiting for the frontend (first start can take a minute)..." -ForegroundColor Cyan
$frontendUp = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Milliseconds 750
    try {
        $null = Invoke-WebRequest "http://localhost:$frontendPort/_stcore/health" -TimeoutSec 3
        Write-Host "  frontend is up" -ForegroundColor Green
        $frontendUp = $true
        break
    } catch { }
}
if (-not $frontendUp) {
    Write-Host "  frontend has not answered yet - it may still be starting." -ForegroundColor Yellow
    Write-Host "  Check its window; the browser will open anyway." -ForegroundColor Yellow
}

if (-not $NoBrowser) {
    Start-Process "http://localhost:$frontendPort"
}

Write-Host ""
Write-Host "  Frontend : http://localhost:$frontendPort"  -ForegroundColor Green
Write-Host "  Backend  : http://127.0.0.1:$backendPort"   -ForegroundColor Green
Write-Host "  API docs : http://127.0.0.1:$backendPort/docs" -ForegroundColor Green
Write-Host "  Logs     : $root\data\logs\" -ForegroundColor Green
Write-Host ""
Write-Host "Stop everything with:  .\run_local.ps1 -Stop" -ForegroundColor DarkGray
