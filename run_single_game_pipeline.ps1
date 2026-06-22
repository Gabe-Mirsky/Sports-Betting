param(
    [string]$KalshiStartDate = "2023-10-01",
    [string]$KalshiEndDate = (Get-Date -Format "yyyy-MM-dd"),
    [switch]$Download,
    [switch]$ForceDownload,
    [switch]$RefreshMarkets,
    [switch]$RefreshCandles,
    [switch]$SkipDashboard
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvSitePackages = Join-Path $ProjectRoot ".venv\Lib\site-packages"

function Test-Python {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $false
    }
    try {
        & $Path --version *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Resolve-Python {
    if (Test-Python $VenvPython) {
        return $VenvPython
    }
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython -and (Test-Python $systemPython.Source)) {
        return $systemPython.Source
    }
    if (Test-Python $BundledPython) {
        if (Test-Path $VenvSitePackages) {
            $env:PYTHONPATH = $VenvSitePackages
        }
        return $BundledPython
    }
    throw "No runnable Python found. Recreate .venv or install Python, then run: python -m pip install -r requirements.txt"
}

$Python = Resolve-Python
$Pipeline = Join-Path $ProjectRoot "scripts\run_single_game_research_pipeline.py"
$Arguments = @(
    $Pipeline,
    "--kalshi-start-date", $KalshiStartDate,
    "--kalshi-end-date", $KalshiEndDate
)

if ($Download) {
    $Arguments += "--download"
}
if ($ForceDownload) {
    $Arguments += "--force-download"
}
if (-not $RefreshMarkets) {
    $Arguments += "--skip-market-pull"
}
if (-not $RefreshCandles) {
    $Arguments += "--skip-candles"
}
if ($SkipDashboard) {
    $Arguments += "--skip-dashboard"
}

Write-Host "Project: $ProjectRoot"
Write-Host "Python:  $Python"
if ($env:PYTHONPATH) {
    Write-Host "PYTHONPATH: $env:PYTHONPATH"
}
Write-Host "Cached mode: market pull=$(-not (-not $RefreshMarkets)), candles=$(-not (-not $RefreshCandles))"

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
