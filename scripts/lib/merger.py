"""结果合并工具 - 合并多GPU输出"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ResultMerger:
    """结果合并器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def merge(
        self,
        split_config,
        gpu_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        合并多个GPU的输出

        Args:
            split_config: 分割配置
            gpu_results: GPU结果列表

        Returns:
            合并统计信息
        """
        logger.info(f"合并 {len(gpu_results)} 个GPU输出...")

        split_data = []
        total_scenes = 0

        for result in gpu_results:
            gpu_output = Path(result["output_path"])
            scenes_file = gpu_output / f"{split_config.name}_scenes.json"

            if not scenes_file.exists():
                logger.warning(f"场景文件不存在: {scenes_file}")
                continue

            with open(scenes_file) as f:
                scenes = json.load(f).get("scenes", [])

            for scene in scenes:
                scene_id = scene.get("scene_id", "")

                # 复制图片
                self._copy_images(gpu_output, scene_id)

                # 保存元数据
                self._save_metadata(
                    scene, scene_id, split_config,
                    result["constraints"].get(scene_id, {})
                )

                # 添加到分割数据
                split_data.append({
                    "scene_id": scene_id,
                    "single_view_image": f"images/single_view/{scene_id}.png",
                    "multi_view_images": [
                        f"images/multi_view/{scene_id}/view_{i}.png"
                        for i in range(4)
                    ],
                    "metadata_path": f"metadata/{scene_id}.json",
                    "n_objects": scene.get("n_objects", 0),
                    "tau": split_config.tau,
                    "split": split_config.name,
                })
                total_scenes += 1

            # 清理临时目录
            try:
                shutil.rmtree(gpu_output)
            except Exception as e:
                logger.warning(f"清理失败: {e}")

        # 保存分割文件
        split_file = self.output_dir / "splits" / f"{split_config.name}.json"
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)

        logger.info(f"合并完成: {total_scenes} 个场景")

        return {
            "n_scenes": total_scenes,
            "n_single_view": total_scenes,
            "n_multi_view": total_scenes * 4,
        }

    def _copy_images(self, gpu_output: Path, scene_id: str):
        """复制图片到最终位置"""
        # 多视角
        src_multi = gpu_output / "multi_view" / scene_id
        dst_multi = self.output_dir / "images" / "multi_view" / scene_id

        if src_multi.exists():
            if dst_multi.exists():
                shutil.rmtree(dst_multi)
            shutil.copytree(src_multi, dst_multi)

        # 单视角
        src_single = gpu_output / "single_view" / f"{scene_id}.png"
        dst_single = self.output_dir / "images" / "single_view" / f"{scene_id}.png"

        if src_single.exists():
            shutil.copy(src_single, dst_single)

    def _save_metadata(self, scene: Dict, scene_id: str, split_config, constraints: Dict):
        """保存场景元数据"""
        metadata = {
            **scene,
            "constraints": constraints,
            "split": split_config.name,
            "tau": split_config.tau,
        }

        metadata_file = self.output_dir / "metadata" / f"{scene_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
