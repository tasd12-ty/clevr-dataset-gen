#!/usr/bin/env python3
"""
VLM 多视角批量约束提取 (Task-2)。

遍历 benchmark 中每个场景的 multi_view 图像（默认 4 张），
一次性发送给 VLM 提取空间约束。多视角可消除单视角深度歧义，
区分 view-invariant（3D 拓扑/大小/距离）和 view-dependent（遮挡/轴向）约束。

支持：
- 断点续传（增量保存 + 跳过已完成场景）
- 可选提供已知物体信息
- 多种 VLM 模型
- 速率限制控制

使用示例：
    # 基本用法
    python -m ordinal_spatial.scripts.extract_vlm_multi \
        --benchmark-dir ./data/benchmark \
        --split test_iid \
        --model openai/gpt-4o

    # 限制数量（测试用）
    python -m ordinal_spatial.scripts.extract_vlm_multi \
        --benchmark-dir ./data/benchmark \
        --split test_iid --max-scenes 10

    # 断点续传
    python -m ordinal_spatial.scripts.extract_vlm_multi \
        --benchmark-dir ./data/benchmark \
        --split test_iid --resume

VLM multi-view batch constraint extraction (Task-2).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_completed(output_file: Path) -> set:
  """加载已完成的 scene_id 集合（用于断点续传）。"""
  if not output_file.exists():
    return set()
  try:
    with open(output_file) as f:
      data = json.load(f)
    return {item["scene_id"] for item in data}
  except (json.JSONDecodeError, KeyError):
    return set()


def main():
  parser = argparse.ArgumentParser(
    description="VLM multi-view constraint extraction (Task-2)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
示例:
  python -m ordinal_spatial.scripts.extract_vlm_multi \\
      -b ./data/benchmark -s test_iid -m openai/gpt-4o

  python -m ordinal_spatial.scripts.extract_vlm_multi \\
      -b ./data/benchmark -s test_iid --resume --delay 1.0

环境变量:
  OPENROUTER_API_KEY    OpenRouter API 密钥
  OPENAI_API_KEY        OpenAI API 密钥（使用 --api-base 切换）
    """,
  )

  parser.add_argument(
    "--benchmark-dir", "-b", required=True,
    help="Benchmark 数据集根目录",
  )
  parser.add_argument(
    "--split", "-s", default="test_iid",
    help="数据集划分 (default: test_iid)",
  )
  parser.add_argument(
    "--output-dir", "-o", default="./results/vlm_multi",
    help="输出目录 (default: ./results/vlm_multi)",
  )

  # VLM 配置
  model_group = parser.add_argument_group("VLM 配置")
  model_group.add_argument(
    "--model", "-m", default="google/gemma-3-27b-it",
    help="VLM 模型名称 (default: google/gemma-3-27b-it)",
  )
  model_group.add_argument(
    "--api-base", default="https://openrouter.ai/api/v1",
    help="API base URL (default: OpenRouter)",
  )
  model_group.add_argument(
    "--api-key", default=None,
    help="API key（默认从 OPENROUTER_API_KEY 环境变量读取）",
  )
  model_group.add_argument(
    "--temperature", type=float, default=0.0,
    help="采样温度 (default: 0.0)",
  )
  model_group.add_argument(
    "--max-tokens", type=int, default=8192,
    help="最大输出 token 数 (default: 8192, 多视角需要更多)",
  )
  model_group.add_argument(
    "--timeout", type=float, default=180.0,
    help="API 超时秒数 (default: 180, 多图推理更慢)",
  )

  # 提取选项
  extract_group = parser.add_argument_group("提取选项")
  extract_group.add_argument(
    "--tau", type=float, default=0.10,
    help="容差参数 (default: 0.10)",
  )
  extract_group.add_argument(
    "--no-objects", action="store_true",
    help="不提供已知物体信息（让 VLM 自行识别）",
  )
  extract_group.add_argument(
    "--max-scenes", type=int, default=None,
    help="最大处理场景数（用于测试）",
  )
  extract_group.add_argument(
    "--save-every", type=int, default=10,
    help="每 N 个场景增量保存一次 (default: 10)",
  )
  extract_group.add_argument(
    "--resume", action="store_true",
    help="断点续传（跳过已完成的场景）",
  )
  extract_group.add_argument(
    "--delay", type=float, default=0.0,
    help="每次 API 调用后的延迟秒数（速率限制）",
  )

  args = parser.parse_args()

  # 延迟导入
  from ordinal_spatial.agents.vlm_constraint_agent import (
    VLMConstraintAgent, VLMAgentConfig,
  )

  benchmark_dir = Path(args.benchmark_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # 加载 split
  split_file = benchmark_dir / "splits" / f"{args.split}.json"
  if not split_file.exists():
    logger.error(f"Split file not found: {split_file}")
    return 1

  with open(split_file) as f:
    split_data = json.load(f)

  if args.max_scenes:
    split_data = split_data[:args.max_scenes]

  logger.info(f"Split '{args.split}': {len(split_data)} scenes")
  logger.info(f"Model: {args.model}")

  # 断点续传
  out_file = output_dir / f"{args.split}_vlm_multi.json"
  completed = set()
  results = []
  if args.resume:
    completed = load_completed(out_file)
    if completed:
      with open(out_file) as f:
        results = json.load(f)
      logger.info(f"Resuming: {len(completed)} scenes already done")

  # 创建 agent
  config = VLMAgentConfig(
    model=args.model,
    api_base=args.api_base,
    api_key=args.api_key,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    timeout=args.timeout,
    retry_count=3,
    validate_consistency=True,
  )
  agent = VLMConstraintAgent(config)

  errors = []
  n_new = 0

  for i, item in enumerate(split_data):
    scene_id = item.get("scene_id", f"scene_{i}")

    # 跳过已完成
    if scene_id in completed:
      continue

    # 多视角图像路径
    mv_rels = item.get("multi_view_images", [])
    if not mv_rels:
      logger.warning(f"  [{i+1}] {scene_id}: no multi_view_images field")
      errors.append({"scene_id": scene_id, "error": "no multi_view_images"})
      continue

    image_paths = [str(benchmark_dir / p) for p in mv_rels]

    # 检查图像是否存在
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
      logger.warning(
        f"  [{i+1}] {scene_id}: {len(missing)} images missing"
      )
      errors.append({
        "scene_id": scene_id,
        "error": f"{len(missing)} images missing",
      })
      continue

    # 可选：加载已知物体信息
    objects = None
    if not args.no_objects:
      meta_rel = item.get("metadata_path", "")
      meta_path = benchmark_dir / meta_rel
      if meta_path.exists():
        with open(meta_path) as f:
          meta = json.load(f)
        objects = meta.get("objects")

    try:
      cs = agent.extract_from_multi_view(
        images=image_paths,
        objects=objects,
        tau=args.tau,
      )
      results.append({"scene_id": scene_id, **cs.to_dict()})
      n_new += 1

      logger.info(
        f"  [{i+1}/{len(split_data)}] {scene_id}: "
        f"{cs.total_constraints()} constraints, "
        f"confidence={cs.confidence:.2f} "
        f"({len(image_paths)} views)"
      )
    except Exception as e:
      logger.warning(f"  [{i+1}] {scene_id}: ERROR {e}")
      errors.append({"scene_id": scene_id, "error": str(e)})

    # 增量保存
    if n_new > 0 and n_new % args.save_every == 0:
      with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
      logger.info(f"  Checkpoint: {len(results)} results saved")

    # 速率限制
    if args.delay > 0:
      time.sleep(args.delay)

  # 最终保存
  with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

  if errors:
    err_file = output_dir / f"{args.split}_vlm_multi_errors.json"
    with open(err_file, "w") as f:
      json.dump(errors, f, indent=2)

  print(f"\n{'=' * 50}")
  print(f"VLM Multi-View Extraction (Task-2)")
  print(f"{'=' * 50}")
  print(f"  Model:   {args.model}")
  print(f"  Split:   {args.split}")
  print(f"  Views:   {len(mv_rels)} per scene")
  print(f"  Success: {len(results)}")
  print(f"  Errors:  {len(errors)}")
  print(f"  Output:  {out_file}")

  return 0


if __name__ == "__main__":
  sys.exit(main())
