"""GPU工作进程 - 单个GPU的渲染任务"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GPUWorker:
    """GPU工作进程"""

    @staticmethod
    def render(task: Dict[str, Any]) -> Dict[str, Any]:
        """
        在单个GPU上执行渲染任务

        Args:
            task: 任务配置字典

        Returns:
            渲染结果统计
        """
        gpu_id = task["gpu_id"]
        split_name = task["split_name"]

        # 设置CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        logger.info(f"GPU {gpu_id}: 开始渲染 {split_name} "
                   f"({task['start_idx']} ~ {task['start_idx'] + task['n_scenes'] - 1})")

        # 准备输出目录
        is_windows = task["blender_path"].endswith('.exe')
        if is_windows:
            blender_out = f"D:/gpu{gpu_id}_{split_name}"
            wsl_out = Path(f"/mnt/d/gpu{gpu_id}_{split_name}")
        else:
            wsl_out = Path(task["output_dir"]) / f"gpu{gpu_id}_temp"
            blender_out = str(wsl_out)

        wsl_out.mkdir(parents=True, exist_ok=True)

        # 构建Blender命令
        cmd = [
            task["blender_path"],
            "--background",
            "--python", "image_generation/render_multiview.py",
            "--",
            "--base_scene_blendfile", "image_generation/data/base_scene_v5.blend",
            "--properties_json", "image_generation/data/properties.json",
            "--shape_dir", "image_generation/data/shapes_v5",
            "--material_dir", "image_generation/data/materials_v5",
            "--output_dir", blender_out,
            "--split", split_name,
            "--num_images", str(task["n_scenes"]),
            "--start_idx", str(task["start_idx"]),
            "--min_objects", str(task["min_objects"]),
            "--max_objects", str(task["max_objects"]),
            "--n_views", "4",
            "--camera_distance", "12.0",
            "--elevation", "30.0",
            "--width", str(task["render_config"]["width"]),
            "--height", str(task["render_config"]["height"]),
            "--render_num_samples", str(task["render_config"]["samples"]),
            "--use_gpu", "1",
            "--render_tile_size", "256",
            "--min_pixels_per_object", "0",
        ]

        # 执行渲染
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 4,
                cwd=Path(__file__).parent.parent.parent
            )

            if result.returncode != 0:
                logger.error(f"GPU {gpu_id}: 渲染失败\n{result.stderr[-500:]}")
                raise RuntimeError(f"GPU {gpu_id} 渲染失败")

            elapsed = time.time() - start_time
            logger.info(f"GPU {gpu_id}: 完成 ({elapsed:.1f}秒, "
                       f"{elapsed/task['n_scenes']:.1f}秒/场景)")

        except subprocess.TimeoutExpired:
            logger.error(f"GPU {gpu_id}: 超时")
            raise

        # 提取约束
        constraints = GPUWorker._extract_constraints(
            wsl_out, split_name, task["tau"], gpu_id
        )

        return {
            "gpu_id": gpu_id,
            "output_path": str(wsl_out),
            "n_scenes": task["n_scenes"],
            "render_time": elapsed,
            "constraints": constraints,
        }

    @staticmethod
    def _extract_constraints(
        output_dir: Path,
        split_name: str,
        tau: float,
        gpu_id: int
    ) -> Dict[str, Dict]:
        """提取真值约束"""
        try:
            from ordinal_spatial.agents import BlenderConstraintAgent
        except ImportError:
            logger.warning(f"GPU {gpu_id}: 无法导入约束提取器")
            return {}

        scenes_file = output_dir / f"{split_name}_scenes.json"
        if not scenes_file.exists():
            logger.warning(f"GPU {gpu_id}: 场景文件不存在")
            return {}

        with open(scenes_file) as f:
            scenes = json.load(f).get("scenes", [])

        agent = BlenderConstraintAgent()
        constraints = {}

        for i, scene in enumerate(scenes):
            scene_id = scene.get("scene_id", "")
            try:
                cs = agent.extract_from_single_view(scene, tau)
                constraints[scene_id] = cs.to_dict()
            except Exception as e:
                logger.warning(f"GPU {gpu_id}: 场景{scene_id}约束提取失败: {e}")
                constraints[scene_id] = scene.get("world_constraints", {})

        logger.info(f"GPU {gpu_id}: 提取了 {len(constraints)} 个约束")
        return constraints
