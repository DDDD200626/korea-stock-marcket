$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$nginxDir = Join-Path $repoRoot "nginx"
$nginxExe = Join-Path $repoRoot "tools\nginx-win\nginx-1.26.3\nginx.exe"
$confName = "nginx.stock-predict.conf"

if (-not (Test-Path $nginxExe)) {
    Write-Error "nginx.exe 없음: $nginxExe"
}

Push-Location $nginxDir
try {
    & $nginxExe -p "$nginxDir\" -c $confName -s quit
} finally {
    Pop-Location
}
Write-Host "nginx quit signaled"
