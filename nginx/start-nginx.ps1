# 저장소 루트에서 nginx 기동 (포트 8080). 먼저: frontend npm run build, 백엔드 :8000
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$nginxDir = Join-Path $repoRoot "nginx"
$nginxExe = Join-Path $repoRoot "tools\nginx-win\nginx-1.26.3\nginx.exe"
$confName = "nginx.stock-predict.conf"
$dist = Join-Path $repoRoot "frontend\dist"

if (-not (Test-Path $nginxExe)) {
    Write-Error "nginx.exe 없음: $nginxExe — tools/nginx-win 설치 여부 확인"
}
if (-not (Test-Path (Join-Path $nginxDir "mime.types"))) {
    Write-Error "nginx/mime.types 없음"
}
if (-not (Test-Path (Join-Path $dist "index.html"))) {
    Write-Warning "frontend/dist/index.html 없음. 실행: cd frontend; npm run build"
}

New-Item -ItemType Directory -Force -Path (Join-Path $nginxDir "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $nginxDir "temp") | Out-Null

Push-Location $nginxDir
try {
    & $nginxExe -t -p "$nginxDir\" -c $confName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $nginxExe -p "$nginxDir\" -c $confName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

Write-Host "OK: http://127.0.0.1:8080  (중지: nginx\stop-nginx.ps1)"
