# AceML Studio - Docker Quick Start Script for Windows
# ======================================================
# This script helps you quickly deploy AceML Studio on Rancher Desktop

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AceML Studio - Docker Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
$dockerRunning = docker ps 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Rancher Desktop and try again." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker is running" -ForegroundColor Green

# Check if .env file exists
if (-Not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env file created" -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANT: Please edit .env file and add your API keys!" -ForegroundColor Yellow
    Write-Host "Press Enter to continue after editing, or Ctrl+C to exit..." -ForegroundColor Yellow
    Read-Host
}

# Build and start
Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker-compose build

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Image built successfully" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "Starting AceML Studio..." -ForegroundColor Yellow
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ AceML Studio started successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Application is running!" -ForegroundColor Green
        Write-Host "  URL: http://localhost:5000" -ForegroundColor White
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Useful commands:" -ForegroundColor Yellow
        Write-Host "  View logs:    docker-compose logs -f" -ForegroundColor White
        Write-Host "  Stop app:     docker-compose down" -ForegroundColor White
        Write-Host "  Restart:      docker-compose restart" -ForegroundColor White
        Write-Host ""
        
        # Wait a moment for container to fully start
        Start-Sleep -Seconds 3
        
        # Open browser
        Write-Host "Opening browser..." -ForegroundColor Yellow
        Start-Process "http://localhost:5000"
    } else {
        Write-Host "✗ Failed to start AceML Studio" -ForegroundColor Red
        Write-Host "Check logs with: docker-compose logs" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ Failed to build image" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
}

Write-Host ""
