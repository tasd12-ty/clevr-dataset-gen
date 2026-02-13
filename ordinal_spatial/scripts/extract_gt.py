#!/usr/bin/env python3
"""
从 benchmark metadata 批量提取 GT 约束 (Task-1)。

从 build_benchmark 生成的 metadata JSON 中读取 3D 坐标，
通过 BlenderConstraintAgent 直接计算所有约束类型。
不需要 VLM、不需要图像、不需要 Blender 运行时。

约束类型：
- QRR: 四元距离比较
- TRR: 三元时钟方向
- Axial: 轴向偏序 (left/right/front/behind/above/below)
- Topology: 拓扑关系 (disjoint/touching/overlapping)
- Size: 大小比较
- Closer: 三元距离排序
- Occlusion: 遮挡关系

使用示例：
    python -m ordinal_spatial.scripts.extract_gt \
        --benchmark-dir ./data/benchmark \
        --split test_iid \
        --output-dir ./results/gt

    python -m ordinal_spatial.scripts.extract_gt \
        --benchmark-dir ./data/benchmark \
        --split all --tau 0.05

Extract ground-truth constraints from benchmark metadata (Task-1).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 可用的数据集划分
ALL_SPLITS = [
  "train", "val", "test_iid", "test_comp", "test_hard",
]


def extract_split(
  benchmark_dir: Path,
  split: str,
  output_dir: Path,
  tau: float,
  disable_types: List[str],
):
  """
  对单个 split 执行 GT 约束提取。

  Extract GT constraints for one split.
  """
  # 延迟导入，避免顶层依赖问题
  from ordinal_spatial.agents.blender_constraint_agent import (
    BlenderConstraintAgent, BlenderAgentConfig,
  )

  split_file = benchmark_dir / "splits" / f"{split}.json"
  if not split_file.exists():
    logger.warning(f"Split file not found: {split_file}, skipping")
    return

  with open(split_file) as f:
    split_data = json.load(f)

  logger.info(f"Split '{split}': {len(split_data)} scenes")

  # 构建配置
  config = BlenderAgentConfig(
    tau=tau,
    extract_qrr="qrr" not in disable_types,
    extract_trr="trr" not in disable_types,
    extract_axial="axial" not in disable_types,
    extract_topology="topology" not in disable_types,
    extract_size="size" not in disable_types,
    extract_closer="closer" not in disable_types,
    extract_occlusion="occlusion" not in disable_types,
    compute_transitive_closure=True,
  )
  agent = BlenderConstraintAgent(config)

  results = []
  errors = []

  for i, item in enumerate(split_data):
    scene_id = item.get("scene_id", f"scene_{i}")
    meta_rel = item.get("metadata_path", "")
    meta_path = benchmark_dir / meta_rel

    if not meta_path.exists():
      logger.warning(f"  [{i+1}] {scene_id}: metadata not found: {meta_rel}")
      errors.append({"scene_id": scene_id, "error": "metadata not found"})
      continue

    try:
      with open(meta_path) as f:
        meta = json.load(f)

      # extract_from_single_view 接受 dict 输入
      cs = agent.extract_from_single_view(image=meta, tau=tau)

      results.append({"scene_id": scene_id, **cs.to_dict()})

      if (i + 1) % 100 == 0 or (i + 1) == len(split_data):
        logger.info(
          f"  [{i+1}/{len(split_data)}] {scene_id}: "
          f"{cs.total_constraints()} constraints"
        )
    except Exception as e:
      logger.warning(f"  [{i+1}] {scene_id}: ERROR {e}")
      errors.append({"scene_id": scene_id, "error": str(e)})

  # 保存结果
  output_dir.mkdir(parents=True, exist_ok=True)

  out_file = output_dir / f"{split}_gt.json"
  with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
  logger.info(f"Saved {len(results)} scenes -> {out_file}")

  if errors:
    err_file = output_dir / f"{split}_gt_errors.json"
    with open(err_file, "w") as f:
      json.dump(errors, f, indent=2)
    logger.warning(f"{len(errors)} errors -> {err_file}")

  return {"split": split, "success": len(results), "errors": len(errors)}


def main():
  parser = argparse.ArgumentParser(
    description="Extract GT constraints from benchmark metadata (Task-1)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例:
  # 提取 test_iid 的 GT 约束
  python -m ordinal_spatial.scripts.extract_gt \\
      --benchmark-dir ./data/benchmark --split test_iid

  # 提取所有 split
  python -m ordinal_spatial.scripts.extract_gt \\
      --benchmark-dir ./data/benchmark --split all

  # 使用严格阈值，只提取 QRR 和 TRR
  python -m ordinal_spatial.scripts.extract_gt \\
      --benchmark-dir ./data/benchmark --split test_hard \\
      --tau 0.05 --disable axial topology size closer occlusion
    """,
  )

  parser.add_argument(
    "--benchmark-dir", "-b", required=True,
    help="Benchmark 数据集根目录",
  )
  parser.add_argument(
    "--split", "-s", default="test_iid",
    help="数据集划分 (test_iid/test_comp/test_hard/train/val/all)",
  )
  parser.add_argument(
    "--output-dir", "-o", default="./results/gt",
    help="输出目录 (default: ./results/gt)",
  )
  parser.add_argument(
    "--tau", type=float, default=0.10,
    help="容差参数 (default: 0.10)",
  )
  parser.add_argument(
    "--disable", nargs="*", default=[],
    choices=["qrr", "trr", "axial", "topology", "size", "closer", "occlusion"],
    help="禁用的约束类型",
  )

  args = parser.parse_args()

  benchmark_dir = Path(args.benchmark_dir)
  output_dir = Path(args.output_dir)

  if not benchmark_dir.exists():
    logger.error(f"Benchmark directory not found: {benchmark_dir}")
    return 1

  # 确定要处理的 split
  if args.split == "all":
    splits = ALL_SPLITS
  else:
    splits = [args.split]

  logger.info(f"Benchmark: {benchmark_dir}")
  logger.info(f"Splits: {splits}")
  logger.info(f"Tau: {args.tau}")
  if args.disable:
    logger.info(f"Disabled types: {args.disable}")

  summary = []
  for split in splits:
    result = extract_split(
      benchmark_dir, split, output_dir, args.tau, args.disable,
    )
    if result:
      summary.append(result)

  # 打印汇总
  print("\n" + "=" * 50)
  print("GT Extraction Summary")
  print("=" * 50)
  for s in summary:
    print(f"  {s['split']}: {s['success']} success, {s['errors']} errors")

  return 0


if __name__ == "__main__":
  sys.exit(main())
