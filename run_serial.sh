#!/bin/bash

# ================= 配置区域 (串行) =================
# 可执行文件路径 (相对于当前目录)
BIN_PATH="./build/bin/hausdorff"
# 模型文件所在目录
MODEL_DIR="./sample_data/model"
# 日志保存目录 (建议修改目录名以区分不同参数的测试结果)
LOG_DIR="./logs_serial_diag_1e-8"

# 模型列表
MODELS=(
    "arm" "bumpy" "camel_b" "elephant" "face" 
    "hand-tri" "inspired_mesh" "lilium" "snail" "truck" 
    "armadillo" "bimba" "happy" "homer" "horse" "cow"
)
# =================================================

# 准备工作
if [ ! -d "$LOG_DIR" ]; then mkdir -p "$LOG_DIR"; fi
echo ">>> 开始运行串行版本测试 (Serial, Error=1e-8, Cond=Diag)"
echo ">>> 结果将保存至: $LOG_DIR"

# 核心测试函数
run_test() {
    local name=$1
    local smooth=$2
    local origin=$3
    
    # 检查文件
    if [[ ! -f "$smooth" || ! -f "$origin" ]]; then
        echo "[跳过] 文件缺失: $name"
        return
    fi

    echo "正在测试: $name ..."
    # 记录文件名
    outfile="${LOG_DIR}/${name}.log"
    
    # 执行命令 (修改点：添加了 -e 1e-8 -c diag 参数)
    # 使用 tee 同时输出到屏幕和文件
    $BIN_PATH -a "$smooth" -b "$origin" -e 1e-8 -c diag -t point | tee "$outfile"
    echo "----------------------------------------"
}

# 循环运行标准模型
for m in "${MODELS[@]}"; do
    run_test "$m" "${MODEL_DIR}/${m}-smooth.obj" "${MODEL_DIR}/${m}.obj"
done

# 单独处理 bunny (文件名可能是 smooth2)
if [ -f "${MODEL_DIR}/bunny-smooth2.obj" ]; then
    run_test "bunny" "${MODEL_DIR}/bunny-smooth2.obj" "${MODEL_DIR}/bunny.obj"
else
    run_test "bunny" "${MODEL_DIR}/bunny-smooth.obj" "${MODEL_DIR}/bunny.obj"
fi

echo "串行测试全部完成。"