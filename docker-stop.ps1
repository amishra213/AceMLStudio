# AceML Studio - Docker Stop Script for Windows
# ===============================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stopping AceML Studio" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ AceML Studio stopped successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Error stopping AceML Studio" -ForegroundColor Red
}

Write-Host ""
