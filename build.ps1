# PPE Detection - Fast Build Script
# TODO : Remove --no-cache for faster rebuilds in subsequent runs

Write-Host "Building PPE Detection Backend..." -ForegroundColor Green
docker-compose build backend --no-cache

Write-Host "Building UI Services..." -ForegroundColor Green
docker-compose build ui edge-device-ui

Write-Host "Build complete! Starting services..." -ForegroundColor Green
docker-compose up -d

Write-Host "Service Status:" -ForegroundColor Yellow
docker-compose ps

Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "  Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "  Main UI:     http://localhost:3000" -ForegroundColor White
Write-Host "  Edge UI:     http://localhost:3001" -ForegroundColor White
Write-Host '  API Docs:    http://localhost:8000/docs' -ForegroundColor White