# Milvus 启动前清理脚本 — Windows 环境
#
# 用途: 清理上次异常退出残留的进程，防止端口/文件锁冲突。
# 在启动 API 服务器或数据导入脚本前运行此脚本。
#
# 用法:
#   powershell -ExecutionPolicy Bypass -File cleanup_milvus.ps1
#   # 或双击运行

Write-Host "=== Milvus 启动前清理 ===" -ForegroundColor Cyan

# ── 1. 清理残留 Python 进程 ──
Write-Host "[1/3] 检查残留 Python 进程 ..."
$pythons = Get-Process -Name "python" -ErrorAction SilentlyContinue
if ($pythons) {
    Write-Host "  发现 $($pythons.Count) 个 Python 进程运行中" -ForegroundColor Yellow
    $pythons | ForEach-Object { Write-Host "    PID=$($_.Id) $($_.MainWindowTitle)" }
    Write-Host "  如需终止，请手动确认后运行: taskkill /F /IM python.exe"
    Write-Host "  (自动终止可能关闭其他不相关任务，已跳过)"
} else {
    Write-Host "  无残留 Python 进程" -ForegroundColor Green
}

# ── 2. 检查 Milvus 端口占用 ──
Write-Host "[2/3] 检查端口 19530 (Milvus) 占用 ..."
$portCheck = netstat -ano | Select-String ":19530"
if ($portCheck) {
    Write-Host "  端口 19530 被占用:" -ForegroundColor Yellow
    Write-Host $portCheck
    Write-Host "  如需释放端口，运行: taskkill /F /PID <PID>"
} else {
    Write-Host "  端口 19530 空闲" -ForegroundColor Green
}

# ── 3. 检查 Docker 状态 ──
Write-Host "[3/3] 检查 Docker 运行状态 ..."
$dockerCheck = docker info 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Docker 运行正常" -ForegroundColor Green
    $milvusRunning = docker ps --filter "name=milvus-standalone" --format "{{.Names}}"
    if ($milvusRunning) {
        Write-Host "  Milvus Standalone 容器已在运行: $milvusRunning" -ForegroundColor Green
        Write-Host ""
        Write-Host "  如需重启 Milvus:" -ForegroundColor Cyan
        Write-Host "    docker compose down; docker compose up -d"
    } else {
        Write-Host "  Milvus Standalone 未运行" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  启动 Milvus 命令:" -ForegroundColor Cyan
        Write-Host "    docker compose up -d"
    }
} else {
    Write-Host "  Docker 未运行或不可用 — 请先启动 Docker Desktop" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== 清理完成 ===" -ForegroundColor Cyan
pause
