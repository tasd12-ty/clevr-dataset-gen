#!/bin/bash
# 8卡GPU数据集生成 - 超简洁入口
#
# 使用方法:
#   ./scripts/generate.sh tiny              # 快速测试（5分钟）
#   ./scripts/generate.sh small             # 小规模（2小时）
#   ./scripts/generate.sh large high        # 完整高质量（200小时）

set -e  # 遇到错误立即退出

# ============= 配置区域（用户只需修改这里）=============
OUTPUT_DIR="./data/benchmark"           # 输出目录
BLENDER_PATH="/mnt/d/tools/blender/blender.exe"  # Blender路径
N_GPUS=8                                # GPU数量
# ======================================================

# 解析参数
SIZE=${1:-small}        # 规模: tiny/small/medium/large
QUALITY=${2:-normal}    # 质量: draft/normal/high

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  ORDINAL-SPATIAL 数据集生成${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "规模:     ${GREEN}${SIZE}${NC}"
echo -e "质量:     ${GREEN}${QUALITY}${NC}"
echo -e "输出:     ${GREEN}${OUTPUT_DIR}_${SIZE}${NC}"
echo -e "GPU数量:  ${GREEN}${N_GPUS}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 运行Python脚本
python scripts/build_8gpu.py \
    --output "${OUTPUT_DIR}_${SIZE}" \
    --size "${SIZE}" \
    --quality "${QUALITY}" \
    --n-gpus "${N_GPUS}" \
    --blender "${BLENDER_PATH}"

echo -e "\n${GREEN}✓ 生成完成!${NC}"
echo -e "输出目录: ${BLUE}${OUTPUT_DIR}_${SIZE}${NC}"
