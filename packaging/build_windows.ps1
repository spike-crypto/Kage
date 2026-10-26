<#
    Build a self-contained Kage for Windows.

    Produces dist\Kage\ holding an embedded Python, the core dependencies, the app and
    a vendored FFmpeg - so a user installs nothing. Roughly 500 MB.

    Deliberately not PyInstaller: torch alone is 4.3 GB and freezing Streamlit, which
    is a server plus a browser, fights the tool rather than using it. An embedded
    interpreter is simpler, debuggable, and the thing it runs is the same launcher.py
    that CI exercises.

    Usage:  powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
#>
param(
    [string]$PythonVersion = "3.12.8",
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $root "dist\Kage"
$work  = Join-Path $root "dist\_work"

Write-Host "Kage packaging" -ForegroundColor Magenta
Write-Host "  root : $root"

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage, $work | Out-Null

# --- 1. embedded Python ------------------------------------------------------
$embedZip = Join-Path $work "python-embed.zip"
if (-not (Test-Path $embedZip)) {
    $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
    Write-Host "  downloading $url" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $embedZip
}
$runtime = Join-Path $stage "runtime"
Expand-Archive -Path $embedZip -DestinationPath $runtime -Force

# The embedded build ships with site-packages disabled; without this pip installs
# nothing importable and the app fails at the first import.
$pth = Get-ChildItem -Path $runtime -Filter "python*._pth" | Select-Object -First 1
(Get-Content $pth.FullName) -replace '^#\s*import site', 'import site' |
    Set-Content $pth.FullName
Add-Content $pth.FullName "`nLib\site-packages`n..`n"

# --- 2. pip, then the core dependencies --------------------------------------
$getPip = Join-Path $work "get-pip.py"
if (-not (Test-Path $getPip)) {
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
}
$py = Join-Path $runtime "python.exe"
Write-Host "  installing pip" -ForegroundColor Cyan
& $py $getPip --no-warn-script-location | Out-Null

Write-Host "  installing dependencies (~322 MB)" -ForegroundColor Cyan
& $py -m pip install --no-warn-script-location -r (Join-Path $PSScriptRoot "requirements-core.txt")
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

# --- 3. the app --------------------------------------------------------------
Write-Host "  copying app" -ForegroundColor Cyan
foreach ($item in @("app.py", "launcher.py", "main.py", "backend", "frontend",
                    "app_pages", ".streamlit", "scripts", "README.md", "LICENSE")) {
    $src = Join-Path $root $item
    if (Test-Path $src) { Copy-Item $src -Destination $stage -Recurse -Force }
}
Get-ChildItem $stage -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# --- 4. FFmpeg ---------------------------------------------------------------
$tools = Join-Path $stage "tools\bin"
New-Item -ItemType Directory -Force -Path $tools | Out-Null
foreach ($exe in @("ffmpeg.exe", "ffprobe.exe")) {
    $src = Join-Path $root "tools\bin\$exe"
    if (Test-Path $src) { Copy-Item $src $tools -Force }
    else { Write-Warning "  $exe not found - the build will not be able to render" }
}

# --- 5. launcher -------------------------------------------------------------
@'
@echo off
title Kage
cd /d "%~dp0"
runtime\python.exe launcher.py
if errorlevel 1 pause
'@ | Set-Content (Join-Path $stage "Kage.bat") -Encoding ASCII

@'
Kage - from ink to imagination

Double-click Kage.bat. The studio opens in your browser at localhost:8501.
Closing the window stops everything.

Nothing to install: Python and FFmpeg are included.

Optional, offered in the app when first needed:
  - Vision models (~1.4 GB) for page selection and depth parallax
  - A free API key for AI narration
'@ | Set-Content (Join-Path $stage "READ ME FIRST.txt") -Encoding UTF8

# --- 6. report ---------------------------------------------------------------
$size = (Get-ChildItem $stage -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("  built dist\Kage  ({0:N0} MB)" -f $size) -ForegroundColor Green

if (-not $SkipZip) {
    $zip = Join-Path $root "dist\Kage-windows-x64.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Write-Host "  compressing..." -ForegroundColor Cyan
    Compress-Archive -Path "$stage\*" -DestinationPath $zip -CompressionLevel Optimal
    $zipSize = (Get-Item $zip).Length / 1MB
    Write-Host ("  {0}  ({1:N0} MB)" -f $zip, $zipSize) -ForegroundColor Green
}
