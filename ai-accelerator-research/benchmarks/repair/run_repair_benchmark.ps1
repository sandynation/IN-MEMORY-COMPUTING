Write-Host "Running Repair Benchmark..." -ForegroundColor Cyan
Set-Location D:\CodexTest
python benchmarks\repair\benchmark_repair.py
Write-Host "Benchmark complete. Results in results/repair_benchmark/" -ForegroundColor Green
