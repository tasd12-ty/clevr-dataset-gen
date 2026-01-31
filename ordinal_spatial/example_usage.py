#!/usr/bin/env python3
"""
ORDINAL-SPATIAL 使用示例

演示如何使用框架的主要功能。

Example usage of ORDINAL-SPATIAL framework.
"""

import numpy as np
from typing import Dict, List


def example_1_basic_comparison():
    """示例 1: 基本比较操作"""
    print("\n" + "="*60)
    print("示例 1: 基本容差比较 | Example 1: Basic Tolerance Comparison")
    print("="*60)

    from ordinal_spatial.dsl.comparators import compare, Comparator

    tau = 0.10  # 10% 容差

    # 测试不同的比较
    test_cases = [
        (10.0, 11.0, "接近相等 | Nearly equal"),
        (8.0, 12.0, "明显小于 | Clearly less"),
        (15.0, 12.0, "明显大于 | Clearly greater"),
    ]

    for a, b, desc in test_cases:
        result = compare(a, b, tau)
        print(f"\ncompare({a}, {b}, tau={tau})")
        print(f"  描述 | Description: {desc}")
        print(f"  结果 | Result: {result.value}")


def example_2_qrr_constraint():
    """示例 2: QRR 约束计算"""
    print("\n" + "="*60)
    print("示例 2: QRR 约束计算 | Example 2: QRR Constraint Computation")
    print("="*60)

    from ordinal_spatial.dsl.predicates import compute_qrr, MetricType

    # 定义物体位置
    objects = {
        "红球 | Red": {"position": [0, 0, 0], "size": 1.0},
        "蓝球 | Blue": {"position": [1, 0, 0], "size": 1.0},
        "绿球 | Green": {"position": [0, 2, 0], "size": 1.0},
        "黄球 | Yellow": {"position": [0, 0, 2], "size": 1.0},
    }

    print("\n物体位置 | Object Positions:")
    for obj_id, obj in objects.items():
        print(f"  {obj_id}: {obj['position']}")

    # 计算约束
    constraint = compute_qrr(
        objects,
        pair1=("红球 | Red", "蓝球 | Blue"),
        pair2=("绿球 | Green", "黄球 | Yellow"),
        metric=MetricType.DIST_3D,
        tau=0.10
    )

    print(f"\n约束 | Constraint: dist(红球,蓝球) {constraint.comparator} dist(绿球,黄球)")
    print(f"比率 | Ratio: {constraint.ratio:.3f}")
    print(f"难度 | Difficulty: {constraint.difficulty}")


def example_3_trr_constraint():
    """示例 3: TRR 时钟约束"""
    print("\n" + "="*60)
    print("示例 3: TRR 时钟约束 | Example 3: TRR Clock Constraint")
    print("="*60)

    from ordinal_spatial.dsl.predicates import compute_trr

    objects = {
        "中心 | Center": {"position": [0, 0, 0]},
        "参考 | Reference": {"position": [1, 0, 0]},  # 向右，12点方向
        "目标 | Target": {"position": [1, 1, 0]},     # 右上，1-2点
    }

    print("\n物体位置 | Object Positions:")
    for obj_id, obj in objects.items():
        print(f"  {obj_id}: {obj['position']}")

    constraint = compute_trr(
        objects,
        target="目标 | Target",
        ref1="中心 | Center",
        ref2="参考 | Reference"
    )

    print(f"\n时钟位置 | Clock Position: {constraint.hour} 点")
    print(f"象限 | Quadrant: {constraint.quadrant}")


def example_4_consistency_check():
    """示例 4: 一致性检查"""
    print("\n" + "="*60)
    print("示例 4: 约束一致性检查 | Example 4: Constraint Consistency")
    print("="*60)

    from ordinal_spatial.evaluation.consistency import check_qrr_consistency

    # 一致的约束集
    consistent_constraints = [
        {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
        {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"},
    ]

    print("\n一致的约束 | Consistent Constraints:")
    for c in consistent_constraints:
        print(f"  dist{c['pair1']} {c['comparator']} dist{c['pair2']}")

    report = check_qrr_consistency(consistent_constraints)
    print(f"结果 | Result: {'✓ 一致 | Consistent' if report.is_consistent else '✗ 不一致 | Inconsistent'}")

    # 不一致的约束集（循环）
    inconsistent_constraints = [
        {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
        {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"},
        {"pair1": ["E", "F"], "pair2": ["A", "B"], "comparator": "<"},  # 形成循环
    ]

    print("\n不一致的约束 | Inconsistent Constraints:")
    for c in inconsistent_constraints:
        print(f"  dist{c['pair1']} {c['comparator']} dist{c['pair2']}")

    report = check_qrr_consistency(inconsistent_constraints)
    print(f"结果 | Result: {'✓ 一致 | Consistent' if report.is_consistent else '✗ 不一致 | Inconsistent'}")
    if not report.is_consistent:
        print(f"发现 {len(report.cycles)} 个矛盾循环 | Found {len(report.cycles)} contradictory cycle(s)")


def example_5_oracle_baseline():
    """示例 5: Oracle 基线评估"""
    print("\n" + "="*60)
    print("示例 5: Oracle 基线 (100%准确率) | Example 5: Oracle Baseline")
    print("="*60)

    from ordinal_spatial.baselines.oracle import OracleBaseline
    from ordinal_spatial.dsl.schema import QRRQuery

    # 创建基线
    baseline = OracleBaseline()

    # 定义场景
    objects = {
        "A": {"position": [0, 0, 0]},
        "B": {"position": [2, 0, 0]},
        "C": {"position": [0, 0, 0]},
        "D": {"position": [1, 0, 0]},
    }

    print("\n物体位置 | Object Positions:")
    for obj_id, obj in objects.items():
        print(f"  {obj_id}: {obj['position']}")

    # 创建查询: dist(A,B) vs dist(C,D)
    query = QRRQuery(
        scene_id="test",
        query_id="q1",
        objects={"A": "A", "B": "B", "C": "C", "D": "D"},
        metric="dist3D",
        tau=0.10
    )

    # 预测
    result = baseline.predict_qrr(objects, query)

    print(f"\n查询 | Query: dist(A,B) vs dist(C,D)")
    print(f"  dist(A,B) = 2.0")
    print(f"  dist(C,D) = 1.0")
    print(f"预测 | Prediction: {result.comparator}")
    print(f"置信度 | Confidence: {result.confidence:.2f}")


def example_6_generate_scene():
    """示例 6: 生成随机场景"""
    print("\n" + "="*60)
    print("示例 6: 生成随机场景 | Example 6: Generate Random Scene")
    print("="*60)

    from ordinal_spatial.scripts.generate_dataset import generate_random_scene

    np.random.seed(42)

    # 生成场景
    scene = generate_random_scene('demo_scene', n_objects=6, tau=0.10)

    print(f"\n场景ID | Scene ID: {scene['scene_id']}")
    print(f"物体数 | Objects: {len(scene['objects'])}")
    print(f"QRR 约束 | QRR Constraints: {len(scene['constraints']['qrr'])}")
    print(f"TRR 约束 | TRR Constraints: {len(scene['constraints']['trr'])}")
    print(f"有效性 | Valid: {'✓' if scene['is_valid'] else '✗'}")

    # 显示几个约束
    print("\n前3个 QRR 约束 | First 3 QRR Constraints:")
    for i, c in enumerate(scene['constraints']['qrr'][:3]):
        print(f"  {i+1}. dist{c['pair1']} {c['comparator']} dist{c['pair2']}")


def example_7_difficulty_levels():
    """示例 7: 难度等级"""
    print("\n" + "="*60)
    print("示例 7: 约束难度等级 | Example 7: Constraint Difficulty Levels")
    print("="*60)

    from ordinal_spatial.dsl.comparators import difficulty_from_ratio

    test_ratios = [3.0, 1.8, 1.4, 1.2, 1.1, 1.02]

    print("\n比率 | Ratio  →  难度等级 | Difficulty Level")
    print("-" * 50)

    for ratio in test_ratios:
        level = difficulty_from_ratio(ratio)
        descriptions = {
            1: "简单 | Easy",
            2: "中等偏易 | Medium-Easy",
            3: "中等 | Medium",
            4: "中等偏难 | Medium-Hard",
            5: "困难 | Hard",
            6: "极难 | Extreme"
        }
        print(f"{ratio:5.2f}  →  Level {level}: {descriptions[level]}")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("ORDINAL-SPATIAL 框架使用示例")
    print("ORDINAL-SPATIAL Framework Usage Examples")
    print("="*60)

    examples = [
        example_1_basic_comparison,
        example_2_qrr_constraint,
        example_3_trr_constraint,
        example_4_consistency_check,
        example_5_oracle_baseline,
        example_6_generate_scene,
        example_7_difficulty_levels,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n示例 {i} 出错 | Example {i} failed: {e}")

    print("\n" + "="*60)
    print("所有示例完成 | All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
