#!/bin/bash
# 单GPU数据集生成脚本
#
# 使用方法:
#   ./scripts/generate_1gpu.sh tiny              # 快速测试（5分钟）
#   ./scripts/generate_1gpu.sh small             # 小规模（16小时）
#   ./scripts/generate_1gpu.sh medium normal     # 中等规模（160小时）

set -e

# ============= 单GPU配置 =============
OUTPUT_DIR="./data/benchmark_1gpu"
BLENDER_PATH="${BLENDER_PATH:-/mnt/d/tools/blender/blender.exe}"
N_GPUS=1                                # 单GPU
GPU_ID=0                                # 使用第0号GPU（可改为0-7）
# ====================================

SIZE=${1:-tiny}         # 默认tiny（单GPU建议用小规模）
QUALITY=${2:-normal}

# Validate Blender path
if [ ! -x "$BLENDER_PATH" ]; then
    if command -v "$BLENDER_PATH" >/dev/null 2>&1; then
        BLENDER_PATH="$(command -v "$BLENDER_PATH")"
    else
        echo "Error: Blender not found. Set BLENDER_PATH=/path/to/blender"
        exit 1
    fi
fi

# 设置只使用指定GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  单GPU数据集生成 (GPU $GPU_ID)${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "规模:     ${GREEN}${SIZE}${NC}"
echo -e "质量:     ${GREEN}${QUALITY}${NC}"
echo -e "输出:     ${GREEN}${OUTPUT_DIR}_${SIZE}${NC}"
echo -e "GPU:      ${GREEN}$GPU_ID${NC} (单卡模式)"
echo -e "${BLUE}========================================${NC}\n"

# 单GPU建议使用较小规模
if [ "$SIZE" = "large" ]; then
    echo -e "${YELLOW}警告: large规模在单GPU上需要约1600小时（67天）${NC}"
    echo -e "${YELLOW}建议使用 small 或 medium 规模${NC}\n"
fi

python scripts/build_8gpu.py \
    --output "${OUTPUT_DIR}_${SIZE}" \
    --size "${SIZE}" \
    --quality "${QUALITY}" \
    --n-gpus 1 \
    --blender "${BLENDER_PATH}"

echo -e "\n${GREEN}✓ 生成完成!${NC}"
echo -e "输出目录: ${BLUE}${OUTPUT_DIR}_${SIZE}${NC}"
