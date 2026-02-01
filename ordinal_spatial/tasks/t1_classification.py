"""
任务 T1：序关系分类运行器。

T1-Q: QRR（四元相对关系）分类
      给定四个物体 A, B, C, D，判断 dist(A,B) 与 dist(C,D) 的大小关系
      输出：< (小于) | ~= (约等于) | > (大于)

T1-C: TRR（三元时钟关系）分类
      给定三个物体 A (目标), B (原点), C (参考方向)
      判断 A 相对于 B->C 轴的时钟位置
      输出：1-12 小时 + 1-4 象限

评估指标：
- T1-Q: 准确率、宏平均F1、序距离误差
- T1-C: 小时准确率、象限准确率、角度误差

Task T1: Ordinal Classification runners.

T1-Q: QRR (Quaternary Relative Relations) classification
T1-C: TRR (Ternary Clock Relations) classification
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from ordinal_spatial.evaluation.metrics import (
    compute_t1_qrr_metrics,
    compute_t1_trr_metrics,
    T1QRRMetrics,
    T1TRRMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class T1Config:
    """Configuration for T1 task runner."""
    tau: float = 0.10
    with_cot: bool = False
    save_predictions: bool = True
    output_dir: Optional[str] = None


@dataclass
class T1Result:
    """Result from T1 evaluation."""
    metrics: Union[T1QRRMetrics, T1TRRMetrics]
    predictions: List[Dict]
    ground_truth: List[Dict]
    config: T1Config
    errors: List[str] = field(default_factory=list)


class T1QRRRunner:
    """
    Runner for T1-Q (QRR classification) task.

    Evaluates a baseline's ability to compare pairwise distances
    between objects in a scene.
    """

    def __init__(self, baseline, config: T1Config = None):
        """
        Initialize T1-Q runner.

        Args:
            baseline: Baseline implementation with predict_qrr method
            config: Task configuration
        """
        self.baseline = baseline
        self.config = config or T1Config()

    def run(
        self,
        dataset: List[Dict],
        images_dir: Optional[str] = None,
    ) -> T1Result:
        """
        Run T1-Q evaluation on dataset.

        Args:
            dataset: List of scene/query data
            images_dir: Directory containing scene images

        Returns:
            T1Result with metrics and predictions
        """
        predictions = []
        ground_truth = []
        errors = []

        for item in dataset:
            scene = item.get("scene", item)
            queries = item.get("qrr_queries", [])

            # Get image if available
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

            for query in queries:
                query_id = query.get("query_id", "")

                try:
                    # Get prediction from baseline
                    if hasattr(self.baseline, "predict_qrr"):
                        if image and hasattr(self.baseline, "config"):
                            # VLM baseline with image
                            pred = self.baseline.predict_qrr(
                                image,
                                objects,
                                (query["A"], query["B"]),
                                (query["C"], query["D"]),
                                metric=query.get("metric", "dist3D"),
                                tau=self.config.tau,
                            )
                        else:
                            # Oracle baseline without image
                            from ordinal_spatial.dsl.schema import QRRQuery
                            q = QRRQuery(
                                scene_id=scene.get("scene_id", ""),
                                query_id=query_id,
                                objects={
                                    "A": query["A"],
                                    "B": query["B"],
                                    "C": query["C"],
                                    "D": query["D"],
                                },
                                metric=query.get("metric", "dist3D"),
                                tau=self.config.tau,
                            )
                            pred_obj = self.baseline.predict_qrr(objects, q)
                            pred = {
                                "comparator": pred_obj.comparator,
                                "confidence": pred_obj.confidence,
                            }
                    else:
                        raise AttributeError("Baseline missing predict_qrr method")

                    predictions.append({
                        "query_id": query_id,
                        "comparator": pred.get("comparator", "~="),
                        "confidence": pred.get("confidence", 0.5),
                    })

                    # Handle ground_truth as string or dict
                    gt = query.get("ground_truth", "~=")
                    gt_comparator = gt.get("comparator", "~=") if isinstance(gt, dict) else gt
                    ground_truth.append({
                        "query_id": query_id,
                        "comparator": gt_comparator,
                    })

                except Exception as e:
                    logger.warning(f"Error on query {query_id}: {e}")
                    errors.append(f"{query_id}: {e}")
                    predictions.append({
                        "query_id": query_id,
                        "comparator": "~=",
                        "confidence": 0.0,
                        "error": str(e),
                    })
                    # Handle ground_truth as string or dict
                    gt = query.get("ground_truth", "~=")
                    gt_comparator = gt.get("comparator", "~=") if isinstance(gt, dict) else gt
                    ground_truth.append({
                        "query_id": query_id,
                        "comparator": gt_comparator,
                    })

        # Compute metrics
        metrics = compute_t1_qrr_metrics(predictions, ground_truth)

        # Save if configured
        if self.config.save_predictions and self.config.output_dir:
            self._save_results(predictions, ground_truth, metrics)

        return T1Result(
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            config=self.config,
            errors=errors,
        )

    def _save_results(self, predictions, ground_truth, metrics):
        """
        保存结果到输出目录。

        Save results to output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)


class T1TRRRunner:
    """
    Runner for T1-C (TRR classification) task.

    Evaluates a baseline's ability to determine clock-face
    directional positions.
    """

    def __init__(self, baseline, config: T1Config = None):
        """
        Initialize T1-C runner.

        Args:
            baseline: Baseline implementation with predict_trr method
            config: Task configuration
        """
        self.baseline = baseline
        self.config = config or T1Config()

    def run(
        self,
        dataset: List[Dict],
        images_dir: Optional[str] = None,
    ) -> T1Result:
        """
        Run T1-C evaluation on dataset.

        Args:
            dataset: List of scene/query data
            images_dir: Directory containing scene images

        Returns:
            T1Result with metrics and predictions
        """
        predictions = []
        ground_truth = []
        errors = []

        for item in dataset:
            scene = item.get("scene", item)
            queries = item.get("trr_queries", [])

            image = None
            if images_dir:
                image_path = item.get("image_path", "")
                if image_path:
                    full_path = Path(images_dir) / image_path
                    if full_path.exists():
                        image = str(full_path)

            objects = {
                obj.get("id", f"obj_{i}"): obj
                for i, obj in enumerate(scene.get("objects", []))
            }

            for query in queries:
                query_id = query.get("query_id", "")

                try:
                    if hasattr(self.baseline, "predict_trr"):
                        if image and hasattr(self.baseline, "config"):
                            pred = self.baseline.predict_trr(
                                image,
                                objects,
                                query["target"],
                                query["ref1"],
                                query["ref2"],
                            )
                        else:
                            from ordinal_spatial.dsl.schema import TRRQuery
                            q = TRRQuery(
                                scene_id=scene.get("scene_id", ""),
                                query_id=query_id,
                                target=query["target"],
                                ref1=query["ref1"],
                                ref2=query["ref2"],
                            )
                            pred_obj = self.baseline.predict_trr(objects, q)
                            pred = {
                                "hour": pred_obj.hour,
                                "quadrant": pred_obj.quadrant,
                                "confidence": pred_obj.confidence,
                            }
                    else:
                        raise AttributeError("Baseline missing predict_trr method")

                    predictions.append({
                        "query_id": query_id,
                        "hour": pred.get("hour", 12),
                        "quadrant": pred.get("quadrant", 1),
                        "confidence": pred.get("confidence", 0.5),
                    })

                    gt = query.get("ground_truth", {})
                    ground_truth.append({
                        "query_id": query_id,
                        "hour": gt.get("hour", 12),
                        "quadrant": gt.get("quadrant", 1),
                    })

                except Exception as e:
                    logger.warning(f"Error on query {query_id}: {e}")
                    errors.append(f"{query_id}: {e}")
                    predictions.append({
                        "query_id": query_id,
                        "hour": 12,
                        "quadrant": 1,
                        "confidence": 0.0,
                        "error": str(e),
                    })
                    gt = query.get("ground_truth", {})
                    ground_truth.append({
                        "query_id": query_id,
                        "hour": gt.get("hour", 12),
                        "quadrant": gt.get("quadrant", 1),
                    })

        metrics = compute_t1_trr_metrics(predictions, ground_truth)

        if self.config.save_predictions and self.config.output_dir:
            self._save_results(predictions, ground_truth, metrics)

        return T1Result(
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            config=self.config,
            errors=errors,
        )

    def _save_results(self, predictions, ground_truth, metrics):
        """
        保存结果到输出目录。

        Save results to output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_t1_qrr_evaluation(
    baseline,
    dataset: List[Dict],
    tau: float = 0.10,
    images_dir: Optional[str] = None,
) -> T1QRRMetrics:
    """
    Quick function to run T1-Q evaluation.

    Args:
        baseline: Baseline implementation
        dataset: Evaluation dataset
        tau: Tolerance parameter
        images_dir: Optional images directory

    Returns:
        T1QRRMetrics
    """
    config = T1Config(tau=tau, save_predictions=False)
    runner = T1QRRRunner(baseline, config)
    result = runner.run(dataset, images_dir)
    return result.metrics


def run_t1_trr_evaluation(
    baseline,
    dataset: List[Dict],
    images_dir: Optional[str] = None,
) -> T1TRRMetrics:
    """
    Quick function to run T1-C evaluation.

    Args:
        baseline: Baseline implementation
        dataset: Evaluation dataset
        images_dir: Optional images directory

    Returns:
        T1TRRMetrics
    """
    config = T1Config(save_predictions=False)
    runner = T1TRRRunner(baseline, config)
    result = runner.run(dataset, images_dir)
    return result.metrics
