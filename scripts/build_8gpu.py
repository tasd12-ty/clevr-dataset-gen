#!/usr/bin/env python3
"""
8卡GPU并行生成 ORDINAL-SPATIAL 数据集 - 用户接口

使用方法：
    python scripts/build_8gpu.py --output ./data/benchmark_full --size large
    python scripts/build_8gpu.py --output ./data/test --size tiny
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.multi_gpu_builder import MultiGPUBuilder, DatasetSize

def main():
    parser = argparse.ArgumentParser(
        description="8卡GPU并行生成数据集 - 简化接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据集规模预设:
  tiny   : 40场景    (测试用，约5分钟)
  small  : 1,520场景 (快速验证，约2小时)
  medium : 15,200场景 (中等规模，约20小时)
  large  : 152,000场景 (完整数据集，约200小时)

示例:
  python scripts/build_8gpu.py --output ./data/test --size tiny
  python scripts/build_8gpu.py --output ./data/full --size large --quality high
        """
    )

    # 用户只需设置这3个参数
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--size",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="数据集规模 (默认: small)"
    )
    parser.add_argument(
        "--quality",
        choices=["draft", "normal", "high"],
        default="normal",
        help="渲染质量 (默认: normal)"
    )

    # 可选高级参数
    default_blender = (
        os.environ.get("BLENDER_PATH")
        or os.environ.get("BLENDER_BIN")
        or "/mnt/d/tools/blender/blender.exe"
    )
    parser.add_argument("--n-gpus", type=int, default=8, help="GPU数量 (默认: 8)")
    parser.add_argument(
        "--blender",
        default=default_blender,
        help="Blender路径 (可用环境变量 BLENDER_PATH/BLENDER_BIN 覆盖)"
    )

    args = parser.parse_args()

    # 打印配置
    print("\n" + "=" * 70)
    print("ORDINAL-SPATIAL 多GPU数据集生成")
    print("=" * 70)
    print(f"输出目录: {args.output}")
    print(f"数据集规模: {args.size}")
    print(f"渲染质量: {args.quality}")
    print(f"GPU数量: {args.n_gpus}")
    print("=" * 70 + "\n")

    # 确认
    response = input("开始生成？(y/N): ")
    if response.lower() != 'y':
        print("已取消")
        return

    # 构建数据集
    builder = MultiGPUBuilder(
        output_dir=args.output,
        blender_path=args.blender,
        n_gpus=args.n_gpus,
        dataset_size=DatasetSize[args.size.upper()],
        quality=args.quality
    )

    builder.build()

if __name__ == "__main__":
    main()
