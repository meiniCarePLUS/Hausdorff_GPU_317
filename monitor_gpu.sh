#!/bin/bash

# ================= GPU 使用情况监控工具 =================
# 用于监控GPU程序运行时的GPU利用率、内存使用和线程并行度

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 命令不可用，无法监控GPU"
    exit 1
fi

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  GPU 设备信息${NC}"
echo -e "${BLUE}======================================${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  实时GPU监控 (每1秒刷新)${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "${YELLOW}按 Ctrl+C 停止监控${NC}"
echo ""

# 实时监控GPU使用情况
watch -n 1 '
echo -e "\033[0;34m========================================\033[0m"
echo -e "\033[0;34m  GPU 使用状态\033[0m"
echo -e "\033[0;34m========================================\033[0m"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | awk -F, '\''{
    printf "GPU利用率:      %s%%\n", $1
    printf "显存利用率:     %s%%\n", $2
    printf "显存使用:       %s / %s MB\n", $3, $4
}'\'''
