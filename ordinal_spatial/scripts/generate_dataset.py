#!/usr/bin/env python3
"""
生成 ORDINAL-SPATIAL 基准测试数据集。

功能：
- 生成随机 3D 场景
- 提取 QRR 和 TRR 约束
- 创建多个数据集划分（训练/验证/测试）
- 控制难度级别
- 验证场景有效性（无退化配置）

数据集划分：
- train: 训练集（10,000 场景）
- val: 验证集（1,500 场景）
- test_iid: 独立同分布测试集（1,500 场景）
- test_comp: 组合泛化测试（500 场景，更多物体）
- test_view: 视角泛化测试（500 场景，新视角）
- test_hard: 困难测试（500 场景，严格容差）

使用示例：
    python generate_dataset.py --small --output-dir ./data
    python generate_dataset.py --n-train 10000 --n-val 1500

Generate ORDINAL-SPATIAL benchmark dataset.

Generates scenes with ordinal constraints for evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    n_train: int = 10000
    n_val: int = 1500
    n_test_iid: int = 1500
    n_test_comp: int = 500
    n_test_view: int = 500
    n_test_hard: int = 500

    # Scene parameters
    min_objects: int = 4
    max_objects: int = 12
    tau: float = 0.10

    # Output
    output_dir: str = "./data/ordinal_spatial"
    random_seed: int = 42


def generate_random_scene(
    scene_id: str,
    n_objects: int,
    tau: float = 0.10,
) -> Dict:
    """
    Generate a random scene with objects and constraints.

    Args:
        scene_id: Unique scene identifier
        n_objects: Number of objects
        tau: Tolerance parameter

    Returns:
        Scene dictionary
    """
    from ordinal_spatial.dsl.predicates import (
        extract_all_qrr,
        extract_all_trr,
        MetricType,
    )
    from ordinal_spatial.generation.degeneracy_checker import is_scene_valid

    # Generate random object positions
    objects = {}
    for i in range(n_objects):
        obj_id = f"obj_{i}"
        position = np.random.uniform(-10, 10, size=3).tolist()
        objects[obj_id] = {
            "id": obj_id,
            "position": position,
            "size": np.random.uniform(0.5, 2.0),
            "color": ["red", "green", "blue", "yellow", "purple", "orange"][i % 6],
        }

    # Extract constraints
    qrr_constraints = extract_all_qrr(objects, tau=tau, metric=MetricType.DIST_3D)
    trr_constraints = extract_all_trr(objects)

    # Check validity - pass objects dict to checker
    valid = is_scene_valid(objects)

    return {
        "scene_id": scene_id,
        "objects": list(objects.values()),
        "constraints": {
            "qrr": [c.to_dict() if hasattr(c, 'to_dict') else asdict(c) for c in qrr_constraints],
            "trr": [c.to_dict() if hasattr(c, 'to_dict') else asdict(c) for c in trr_constraints],
        },
        "tau": tau,
        "is_valid": valid,
    }


def generate_qrr_queries(scene: Dict, n_queries: int = 10) -> List[Dict]:
    """
    为场景生成 QRR 查询。

    Generate QRR queries for a scene.
    """
    objects = scene.get("objects", [])
    obj_ids = [obj["id"] for obj in objects]

    if len(obj_ids) < 4:
        return []

    queries = []
    qrr_constraints = scene.get("constraints", {}).get("qrr", [])

    # Use existing constraints as queries
    for i, constraint in enumerate(qrr_constraints[:n_queries]):
        pair1 = constraint.get("pair1", [])
        pair2 = constraint.get("pair2", [])

        if len(pair1) >= 2 and len(pair2) >= 2:
            queries.append({
                "query_id": f"{scene['scene_id']}_qrr_{i}",
                "A": pair1[0],
                "B": pair1[1],
                "C": pair2[0],
                "D": pair2[1],
                "metric": constraint.get("metric", "dist3D"),
                "ground_truth": {
                    "comparator": constraint.get("comparator", "~="),
                },
            })

    return queries


def generate_trr_queries(scene: Dict, n_queries: int = 10) -> List[Dict]:
    """
    为场景生成 TRR 查询。

    Generate TRR queries for a scene.
    """
    trr_constraints = scene.get("constraints", {}).get("trr", [])

    queries = []
    for i, constraint in enumerate(trr_constraints[:n_queries]):
        queries.append({
            "query_id": f"{scene['scene_id']}_trr_{i}",
            "target": constraint.get("target", ""),
            "ref1": constraint.get("ref1", ""),
            "ref2": constraint.get("ref2", ""),
            "ground_truth": {
                "hour": constraint.get("hour", 12),
                "quadrant": constraint.get("quadrant", 1),
            },
        })

    return queries


def generate_split(
    split_name: str,
    n_scenes: int,
    config: DatasetConfig,
    start_idx: int = 0,
) -> List[Dict]:
    """
    生成数据集分割。

    Generate a dataset split.
    """
    logger.info(f"Generating {split_name} split: {n_scenes} scenes")

    scenes = []
    for i in range(n_scenes):
        scene_id = f"{split_name}_{start_idx + i:06d}"

        # Vary object count
        if split_name == "test_comp":
            # Compositional: higher object counts
            n_objects = np.random.randint(13, 17)
        else:
            n_objects = np.random.randint(
                config.min_objects,
                config.max_objects + 1
            )

        # Vary tau for hard split
        tau = config.tau
        if split_name == "test_hard":
            tau = 0.03  # Stricter tolerance

        scene = generate_random_scene(scene_id, n_objects, tau)

        # Add queries
        scene["qrr_queries"] = generate_qrr_queries(scene)
        scene["trr_queries"] = generate_trr_queries(scene)

        scenes.append(scene)

        if (i + 1) % 100 == 0:
            logger.info(f"  Generated {i + 1}/{n_scenes} scenes")

    return scenes


def generate_full_dataset(config: DatasetConfig) -> Dict[str, List[Dict]]:
    """
    生成包含所有分割的完整数据集。

    Generate complete dataset with all splits.
    """
    np.random.seed(config.random_seed)

    dataset = {}

    # Training
    dataset["train"] = generate_split("train", config.n_train, config)

    # Validation
    dataset["val"] = generate_split("val", config.n_val, config)

    # Test IID
    dataset["test_iid"] = generate_split("test_iid", config.n_test_iid, config)

    # Test compositional (novel object counts)
    dataset["test_comp"] = generate_split("test_comp", config.n_test_comp, config)

    # Test viewpoint (handled by render variations)
    dataset["test_view"] = generate_split("test_view", config.n_test_view, config)

    # Test hard (stricter tolerance)
    dataset["test_hard"] = generate_split("test_hard", config.n_test_hard, config)

    return dataset


def save_dataset(dataset: Dict[str, List[Dict]], output_dir: str):
    """
    保存数据集到磁盘。

    Save dataset to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits_dir = output_path / "splits"
    splits_dir.mkdir(exist_ok=True)

    stats = {"total_scenes": 0, "splits": {}}

    for split_name, scenes in dataset.items():
        split_file = splits_dir / f"{split_name}.json"
        with open(split_file, "w") as f:
            json.dump(scenes, f, indent=2)

        stats["splits"][split_name] = len(scenes)
        stats["total_scenes"] += len(scenes)

        logger.info(f"Saved {split_name}: {len(scenes)} scenes -> {split_file}")

    # Save stats
    with open(output_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Total scenes: {stats['total_scenes']}")


def main():
    """
    主入口点。

    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Generate ORDINAL-SPATIAL benchmark dataset"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./data/ordinal_spatial",
        help="Output directory"
    )
    parser.add_argument(
        "--n-train", type=int, default=1000,
        help="Number of training scenes"
    )
    parser.add_argument(
        "--n-val", type=int, default=150,
        help="Number of validation scenes"
    )
    parser.add_argument(
        "--n-test", type=int, default=150,
        help="Number of test scenes per split"
    )
    parser.add_argument(
        "--min-objects", type=int, default=4,
        help="Minimum objects per scene"
    )
    parser.add_argument(
        "--max-objects", type=int, default=12,
        help="Maximum objects per scene"
    )
    parser.add_argument(
        "--tau", type=float, default=0.10,
        help="Tolerance parameter"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Generate small dataset for testing"
    )

    args = parser.parse_args()

    if args.small:
        config = DatasetConfig(
            n_train=100,
            n_val=20,
            n_test_iid=20,
            n_test_comp=10,
            n_test_view=10,
            n_test_hard=10,
            min_objects=args.min_objects,
            max_objects=args.max_objects,
            tau=args.tau,
            output_dir=args.output_dir,
            random_seed=args.seed,
        )
    else:
        config = DatasetConfig(
            n_train=args.n_train,
            n_val=args.n_val,
            n_test_iid=args.n_test,
            n_test_comp=args.n_test // 3,
            n_test_view=args.n_test // 3,
            n_test_hard=args.n_test // 3,
            min_objects=args.min_objects,
            max_objects=args.max_objects,
            tau=args.tau,
            output_dir=args.output_dir,
            random_seed=args.seed,
        )

    logger.info("Starting dataset generation")
    logger.info(f"Config: {asdict(config)}")

    dataset = generate_full_dataset(config)
    save_dataset(dataset, config.output_dir)

    logger.info("Dataset generation complete!")


if __name__ == "__main__":
    main()
