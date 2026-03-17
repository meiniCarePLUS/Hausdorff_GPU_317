#!/bin/bash

# ================= GPU vs CPU 性能对比测试脚本 =================
# 对比 CPU 串行版本和 GPU 加速版本的性能和正确性
# 说明：
# 1) 当前脚本按你现有 CPU 日志口径统一参数：-e 0.01 -c rel -t point
# 2) CPU 日志目录为 ./log_serial ，其中每个模型一个 log，例如 arm.log
# 3) 比较 [distance] 行时使用上界 U 作为最终距离值

set -u

# ---------------- 配置区域 ----------------
GPU_BIN="./build_gpu/bin/hausdorff"
MODEL_DIR="./sample_data/model"
LOG_DIR="./logs_gpu_comparison"
CPU_RESULTS_DIR="./logs_serial"

# 与现有 CPU log 一致的参数
ERROR_BOUND="0.01"
STOP_CONDITION="rel"
TRAIT_TYPE="point"

# 测试模型列表
MODELS=(
    "arm" "bumpy" "camel_b" "elephant" "face"
    "hand-tri" "inspired_mesh" "lilium" "snail" "truck"
    "armadillo" "bimba" "happy" "homer" "horse" "cow"
    "beast"
)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ---------------- 工具函数 ----------------

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo -e "${RED}错误: 未找到命令 '$cmd'${NC}"
        exit 1
    fi
}

trim() {
    local s="$1"
    # shellcheck disable=SC2001
    echo "$(echo "$s" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
}

extract_field_value() {
    # 用法: extract_field_value logfile "[total_cost]"
    local logfile="$1"
    local key="$2"
    # 转义特殊字符 [ 和 ]
    local escaped_key="${key//\[/\\[}"
    escaped_key="${escaped_key//\]/\\]}"
    grep -m1 "^${escaped_key}" "$logfile" 2>/dev/null | awk '{print $2}'
}

extract_distance_upper() {
    # 提取 [distance] L - U 中的 U
    local logfile="$1"
    grep -m1 "^\\[distance\\]" "$logfile" 2>/dev/null | awk '{print $4}'
}

extract_distance_full() {
    local logfile="$1"
    grep -m1 "^\\[distance\\]" "$logfile" 2>/dev/null | awk '{print $2, $3, $4}'
}

extract_max_point() {
    local logfile="$1"
    grep -m1 "^\\[max_point\\]" "$logfile" 2>/dev/null | awk '{print $2, $3, $4}'
}

safe_divide() {
    local a="$1"
    local b="$2"
    if [ -z "$a" ] || [ -z "$b" ]; then
        echo ""
        return
    fi
    if [ "$(echo "$b == 0" | bc -l)" -eq 1 ]; then
        echo ""
        return
    fi
    echo "$(echo "scale=4; $a / $b" | bc -l)"
}

safe_abs_rel_diff() {
    # |a-b|/|a|, 若 a=0，则退化为 |b|
    local a="$1"
    local b="$2"

    if [ -z "$a" ] || [ -z "$b" ]; then
        echo ""
        return
    fi

    if [ "$(echo "$a == 0" | bc -l)" -eq 1 ]; then
        echo "$(echo "scale=10; sqrt(($b)*($b))" | bc -l)"
    else
        echo "$(echo "scale=10; sqrt((($a)-($b))*((($a)-($b)))) / sqrt(($a)*($a))" | bc -l)"
    fi
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  GPU vs CPU Hausdorff 距离计算对比测试${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "测试参数: -e ${ERROR_BOUND} -c ${STOP_CONDITION} -t ${TRAIT_TYPE}"
    echo "CPU 日志目录: ${CPU_RESULTS_DIR}"
    echo "GPU 日志目录: ${LOG_DIR}"
    echo ""
}

# ---------------- 前置检查 ----------------

require_command grep
require_command awk
require_command bc

if [ ! -f "$GPU_BIN" ]; then
    echo -e "${RED}错误: GPU版本可执行文件不存在: $GPU_BIN${NC}"
    echo -e "${YELLOW}提示: 请先编译 GPU 版本: cmake -S . -B build_gpu && cmake --build build_gpu${NC}"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}错误: 模型目录不存在: $MODEL_DIR${NC}"
    exit 1
fi

if [ ! -d "$CPU_RESULTS_DIR" ]; then
    echo -e "${RED}错误: CPU 日志目录不存在: $CPU_RESULTS_DIR${NC}"
    exit 1
fi

mkdir -p "$LOG_DIR"

print_header

# ---------------- 汇总文件 ----------------

SUMMARY_FILE="${LOG_DIR}/summary.csv"
{
    echo "Model,CPU_Time_ms,GPU_Time_ms,Speedup,CPU_Distance_U,GPU_Distance_U,Rel_Diff,Distance_Match"
} > "$SUMMARY_FILE"

# ---------------- 测试函数 ----------------

run_comparison() {
    local name="$1"
    local smooth=""
    local origin=""
    local cpu_log="${CPU_RESULTS_DIR}/${name}.log"
    local gpu_log="${LOG_DIR}/${name}_gpu.log"

    # 处理 bunny 特例
    if [ "$name" = "bunny" ]; then
        if [ -f "${MODEL_DIR}/bunny-smooth2.obj" ]; then
            smooth="${MODEL_DIR}/bunny-smooth2.obj"
        elif [ -f "${MODEL_DIR}/bunny-smooth.obj" ]; then
            smooth="${MODEL_DIR}/bunny-smooth.obj"
        else
            echo -e "${YELLOW}[跳过] 缺少 bunny 的 smooth 模型${NC}"
            return
        fi
        origin="${MODEL_DIR}/bunny.obj"
    else
        smooth="${MODEL_DIR}/${name}-smooth.obj"
        origin="${MODEL_DIR}/${name}.obj"
    fi

    if [[ ! -f "$smooth" || ! -f "$origin" ]]; then
        echo -e "${YELLOW}[跳过] 模型文件缺失: $name${NC}"
        echo "  smooth: $smooth"
        echo "  origin: $origin"
        return
    fi

    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}测试模型: $name${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # ---------- 读取 CPU 日志 ----------
    local cpu_time=""
    local cpu_distance=""
    local cpu_distance_u=""
    local cpu_max_point=""
    local cpu_mean=""
    local cpu_stop=""
    local cpu_err=""

    if [ -f "$cpu_log" ]; then
        echo -e "${BLUE}[1/2] 读取 CPU 串行版本结果...${NC}"

        cpu_time="$(trim "$(extract_field_value "$cpu_log" "[total_cost]")")"
        cpu_distance="$(trim "$(extract_distance_full "$cpu_log")")"
        cpu_distance_u="$(trim "$(extract_distance_upper "$cpu_log")")"
        cpu_max_point="$(trim "$(extract_max_point "$cpu_log")")"
        cpu_mean="$(trim "$(extract_field_value "$cpu_log" "[mean_distance]")")"
        cpu_stop="$(trim "$(extract_field_value "$cpu_log" "[stop_condition]")")"
        cpu_err="$(trim "$(extract_field_value "$cpu_log" "[rel_error_bound]")")"

        echo "  停止条件: ${cpu_stop:-N/A}"
        echo "  误差界:   ${cpu_err:-N/A}"
        echo "  耗时:     ${cpu_time:-N/A} ms"
        echo "  距离:     ${cpu_distance:-N/A}"
        echo "  最大点:   ${cpu_max_point:-N/A}"
        echo "  平均距离: ${cpu_mean:-N/A}"

        # 参数口径提醒
        if [ "$cpu_stop" != "$STOP_CONDITION" ]; then
            echo -e "${YELLOW}  警告: CPU 日志 stop_condition=${cpu_stop}，与当前脚本 ${STOP_CONDITION} 不一致${NC}"
        fi
        if [ "$cpu_err" != "$ERROR_BOUND" ]; then
            echo -e "${YELLOW}  警告: CPU 日志 rel_error_bound=${cpu_err}，与当前脚本 ${ERROR_BOUND} 不一致${NC}"
        fi
    else
        echo -e "${YELLOW}[1/2] CPU 日志不存在，跳过 CPU 对比: $cpu_log${NC}"
    fi
    echo ""

    # ---------- 运行 GPU ----------
    echo -e "${BLUE}[2/2] 运行 GPU 加速版本...${NC}"
    "$GPU_BIN" \
        -a "$smooth" \
        -b "$origin" \
        -e "$ERROR_BOUND" \
        -c "$STOP_CONDITION" \
        -t "$TRAIT_TYPE" \
        > "$gpu_log" 2>&1

    local gpu_exit_code=$?
    if [ $gpu_exit_code -ne 0 ]; then
        echo -e "${RED}错误: GPU 程序执行失败，退出码: $gpu_exit_code${NC}"
        echo -e "${YELLOW}请检查日志: $gpu_log${NC}"
        echo "$name,${cpu_time:-},,,${cpu_distance_u:-},,,GPU_RUN_FAIL" >> "$SUMMARY_FILE"
        echo ""
        return
    fi

    local gpu_time=""
    local gpu_distance=""
    local gpu_distance_u=""
    local gpu_max_point=""
    local gpu_mean=""

    gpu_time="$(trim "$(extract_field_value "$gpu_log" "[total_cost]")")"
    gpu_distance="$(trim "$(extract_distance_full "$gpu_log")")"
    gpu_distance_u="$(trim "$(extract_distance_upper "$gpu_log")")"
    gpu_max_point="$(trim "$(extract_max_point "$gpu_log")")"
    gpu_mean="$(trim "$(extract_field_value "$gpu_log" "[mean_distance]")")"

    if [ -z "$gpu_time" ]; then
        echo -e "${RED}错误: 无法从 GPU 日志提取 [total_cost]${NC}"
        echo -e "${YELLOW}请检查日志: $gpu_log${NC}"
        echo "$name,${cpu_time:-},,,${cpu_distance_u:-},,,GPU_PARSE_FAIL" >> "$SUMMARY_FILE"
        echo ""
        return
    fi

    echo "  耗时:     ${gpu_time} ms"
    echo "  距离:     ${gpu_distance:-N/A}"
    echo "  最大点:   ${gpu_max_point:-N/A}"
    echo "  平均距离: ${gpu_mean:-N/A}"
    echo ""

    # ---------- 计算加速比 ----------
    local speedup=""
    if [ -n "$cpu_time" ]; then
        speedup="$(safe_divide "$cpu_time" "$gpu_time")"
        if [ -n "$speedup" ]; then
            echo -e "${YELLOW}加速比: ${speedup}x${NC}"
        else
            echo -e "${YELLOW}加速比: N/A${NC}"
        fi
    else
        speedup="N/A"
        echo -e "${YELLOW}加速比: N/A（无 CPU 时间）${NC}"
    fi

    # ---------- 验证结果一致性 ----------
    local match="N/A"
    local rel_diff=""

    if [ -n "$cpu_distance_u" ] && [ -n "$gpu_distance_u" ]; then
        rel_diff="$(safe_abs_rel_diff "$cpu_distance_u" "$gpu_distance_u")"
        local rel_diff_pct=""
        rel_diff_pct="$(echo "scale=6; $rel_diff * 100" | bc -l)"

        # 阈值：0.1%
        if [ "$(echo "$rel_diff < 0.001" | bc -l)" -eq 1 ]; then
            echo -e "${GREEN}✓ 结果验证: 通过 (相对误差: ${rel_diff_pct}%)${NC}"
            match="PASS"
        else
            echo -e "${RED}✗ 结果验证: 失败 (相对误差: ${rel_diff_pct}%)${NC}"
            match="FAIL"
        fi
    else
        echo -e "${YELLOW}结果验证: N/A（缺少 CPU 或 GPU distance）${NC}"
        match="N/A"
        rel_diff=""
    fi

    echo "$name,${cpu_time:-},${gpu_time:-},${speedup:-},${cpu_distance_u:-},${gpu_distance_u:-},${rel_diff:-},${match}" >> "$SUMMARY_FILE"
    echo ""
}

# ---------------- 主流程 ----------------

for model in "${MODELS[@]}"; do
    run_comparison "$model"
done

# 如果模型目录中存在 bunny，再补跑
if [ -f "${MODEL_DIR}/bunny.obj" ]; then
    run_comparison "bunny"
fi

# ---------------- 显示汇总 ----------------

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  测试汇总${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if command -v column >/dev/null 2>&1; then
    column -t -s',' "$SUMMARY_FILE"
else
    cat "$SUMMARY_FILE"
fi

echo ""
echo -e "${GREEN}详细日志保存在: $LOG_DIR${NC}"
echo -e "${GREEN}汇总文件: $SUMMARY_FILE${NC}"