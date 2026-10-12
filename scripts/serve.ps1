# Run the lunar plate solver API on this PC and publish it through a Cloudflare Tunnel.
#
#   powershell -ExecutionPolicy Bypass -File scripts\serve.ps1          # named tunnel (token in deploy\server\.env)
#   powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 -Quick   # temporary trycloudflare.com URL, no account
#
# The API listens on 127.0.0.1:8000 only; the tunnel is the one way in from outside.
# Ctrl+C stops the tunnel and the API.
param(
    [switch]$Quick,
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$EnvFile = Join-Path $Root "deploy\server\.env"
$Log = Join-Path $Root "deploy\server\api.log"

if (-not (Test-Path $Python)) { throw "No virtual environment at $Python (see README section 16)." }
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    throw "cloudflared is not installed. Install it with:  winget install --id Cloudflare.cloudflared  (then open a new terminal)"
}

$Token = $null
if (-not $Quick) {
    if (Test-Path $EnvFile) {
        foreach ($line in Get-Content $EnvFile) {
            if ($line -match '^\s*TUNNEL_TOKEN\s*=\s*(.+?)\s*$') { $Token = $Matches[1] }
        }
    }
    if (-not $Token) { throw "Set TUNNEL_TOKEN in $EnvFile (copy deploy\server\.env.example), or use -Quick." }
}

Write-Host "Starting the API on http://127.0.0.1:$Port (log: $Log) ..."
$api = Start-Process -FilePath $Python -ArgumentList @("server\api.py", "--host", "127.0.0.1", "--port", "$Port") `
    -WorkingDirectory $Root -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -PassThru -WindowStyle Hidden
try {
    $ready = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        if ($api.HasExited) { throw "The API stopped. See $Log.err" }
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 3
            if ($h.loaded) { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) { throw "The API did not become ready in 3 minutes. See $Log.err" }
    Write-Host "API ready. Starting the tunnel (Ctrl+C to stop) ..."
    if ($Quick) {
        Write-Host "Look for the https://....trycloudflare.com line below; open the website with ?api=<that URL>."
        & cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$Port"
    } else {
        & cloudflared tunnel --no-autoupdate run --token $Token
    }
} finally {
    if (-not $api.HasExited) {
        Write-Host "Stopping the API ..."
        Stop-Process -Id $api.Id -Force
    }
}
