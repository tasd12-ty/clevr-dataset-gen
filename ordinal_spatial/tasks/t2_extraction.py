"""
任务 T2：约束提取运行器。

评估基线从图像中提取完整、一致的序空间约束集的能力。

任务描述：
- 输入：场景图像 + 物体列表
- 输出：完整的 QRR 和 TRR 约束集
- 目标：提取所有或大部分约束，且保持逻辑一致性

评估指标：
- 精确率（Precision）：预测约束中正确的比例
- 召回率（Recall）：真值约束中被发现的比例
- F1 分数：精确率和召回率的调和平均
- 一致性：约束集是否无矛盾
- 调整后 F1：F1 × 一致性

一致性检查：
- 使用图循环检测算法
- 发现传递性违反（如 A<B<C<A）
- 统计一致和不一致的样本数

Task T2: Constraint Extraction runner.

Evaluates a baseline's ability to extract complete, consistent
sets of ordinal spatial constraints from images.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from ordinal_spatial.evaluation.metrics import compute_t2_metrics, T2Metrics
from ordinal_spatial.evaluation.consistency import check_qrr_consistency

logger = logging.getLogger(__name__)


@dataclass
class T2Config:
    """Configuration for T2 task runner."""
    tau: float = 0.10
    with_cot: bool = False
    check_consistency: bool = True
    save_predictions: bool = True
    output_dir: Optional[str] = None


@dataclass
class T2Result:
    """Result from T2 evaluation."""
    metrics: T2Metrics
    predictions: List[Dict]
    ground_truth: List[Dict]
    consistency_stats: Dict[str, Any]
    config: T2Config
    errors: List[str] = field(default_factory=list)


class T2Runner:
    """
    Runner for T2 (Constraint Extraction) task.

    Evaluates a baseline's ability to extract complete and
    consistent constraint sets from scene images.
    """

    def __init__(self, baseline, config: T2Config = None):
        """
        Initialize T2 runner.

        Args:
            baseline: Baseline with extract_constraints method
            config: Task configuration
        """
        self.baseline = baseline
        self.config = config or T2Config()

    def run(
        self,
        dataset: List[Dict],
        images_dir: Optional[str] = None,
    ) -> T2Result:
        """
        Run T2 evaluation on dataset.

        Args:
            dataset: List of scene data with ground truth
            images_dir: Directory containing scene images

        Returns:
            T2Result with metrics and predictions
        """
        all_predictions = []
        all_ground_truth = []
        errors = []

        consistency_results = {
            "n_consistent": 0,
            "n_inconsistent": 0,
            "n_cycles_total": 0,
        }

        for item in dataset:
            scene = item.get("scene", item)
            scene_id = scene.get("scene_id", "unknown")

            # Get image
            image = None
            if images_dir:
                image_path = item.get("image_path", "")
                if image_path:
                    full_path = Path(images_dir) / image_path
                    if full_path.exists():
                        image = str(full_path)

            # Build objects dict
            objects = {
                obj.get("id", f"obj_{i}"): obj
                for i, obj in enumerate(scene.get("objects", []))
            }

            try:
                # Get prediction from baseline
                if hasattr(self.baseline, "extract_constraints"):
                    if image:
                        pred = self.baseline.extract_constraints(
                            image, objects, tau=self.config.tau
                        )
                    else:
                        # Oracle baseline
                        pred = self.baseline.extract_osd(scene)
                        if hasattr(pred, "model_dump"):
                            pred = pred.model_dump()
                        elif hasattr(pred, "to_dict"):
                            pred = pred.to_dict()
                elif hasattr(self.baseline, "extract_osd"):
                    pred = self.baseline.extract_osd(scene)
                    if hasattr(pred, "model_dump"):
                        pred = pred.model_dump()
                else:
                    raise AttributeError("Baseline missing extraction method")

                # Check consistency
                if self.config.check_consistency:
                    qrr = pred.get("qrr", [])
                    report = check_qrr_consistency(qrr)

                    if report.is_consistent:
                        consistency_results["n_consistent"] += 1
                    else:
                        consistency_results["n_inconsistent"] += 1
                        consistency_results["n_cycles_total"] += len(report.cycles)

                    pred["consistency"] = {
                        "is_consistent": report.is_consistent,
                        "n_cycles": len(report.cycles),
                    }

                all_predictions.append({
                    "scene_id": scene_id,
                    **pred,
                })

            except Exception as e:
                logger.warning(f"Error on scene {scene_id}: {e}")
                errors.append(f"{scene_id}: {e}")
                all_predictions.append({
                    "scene_id": scene_id,
                    "qrr": [],
                    "trr": [],
                    "error": str(e),
                })

            # Get ground truth
            gt = item.get("ground_truth", item.get("constraints", {}))
            all_ground_truth.append({
                "scene_id": scene_id,
                "qrr": gt.get("qrr", scene.get("constraints", {}).get("qrr", [])),
                "trr": gt.get("trr", scene.get("constraints", {}).get("trr", [])),
            })

        # Compute aggregated metrics
        metrics = self._compute_batch_metrics(all_predictions, all_ground_truth)

        # Save if configured
        if self.config.save_predictions and self.config.output_dir:
            self._save_results(all_predictions, all_ground_truth, metrics)

        return T2Result(
            metrics=metrics,
            predictions=all_predictions,
            ground_truth=all_ground_truth,
            consistency_stats=consistency_results,
            config=self.config,
            errors=errors,
        )

    def _compute_batch_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> T2Metrics:
        """
        计算所有场景的指标。

        Compute metrics across all scenes.
        """
        from ordinal_spatial.evaluation.metrics import compute_t2_metrics_batch

        return compute_t2_metrics_batch(predictions, ground_truth)

    def _save_results(self, predictions, ground_truth, metrics):
        """
        保存结果到输出目录。

        Save results to output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        with open(output_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)


def run_t2_evaluation(
    baseline,
    dataset: List[Dict],
    tau: float = 0.10,
    images_dir: Optional[str] = None,
) -> T2Metrics:
    """
    Quick function to run T2 evaluation.

    Args:
        baseline: Baseline implementation
        dataset: Evaluation dataset
        tau: Tolerance parameter
        images_dir: Optional images directory

    Returns:
        T2Metrics
    """
    config = T2Config(tau=tau, save_predictions=False)
    runner = T2Runner(baseline, config)
    result = runner.run(dataset, images_dir)
    return result.metrics
