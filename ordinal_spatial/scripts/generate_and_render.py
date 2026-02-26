#!/usr/bin/env python3
"""
生成场景数据 + 调用 Blender 渲染图片的一体化脚本。

3-10 个物体，每种数量各 10 个场景，共 80 个场景。
每个场景渲染 1 张单视角 + 4 张多视角图片。

用法:
  uv run python -m ordinal_spatial.scripts.generate_and_render \
    --output-dir ./data \
    --blender-path /Applications/Blender.app/Contents/MacOS/Blender

All-in-one: generate scenes + render with Blender.
"""

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from ordinal_spatial.dsl.predicates import MetricType
from ordinal_spatial.generation.constraint_extractor import (
  ConstraintExtractor,
  ExtractionConfig,
)

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# CLEVR 兼容属性
SHAPES = ["cube", "sphere", "cylinder"]
COLORS = [
  "gray", "red", "blue", "green",
  "brown", "purple", "cyan", "yellow",
]
MATERIALS = ["rubber", "metal"]
SIZES = {"large": 0.7, "small": 0.35}


def resolve_image_generation_dir(explicit_dir: str = "") -> Path:
  """
  解析 image_generation 资源目录。

  优先级:
  1) --img-gen-dir 显式参数
  2) 环境变量 IMG_GEN_DIR
  3) ordinal_spatial/image_generation（打包内置）
  4) 仓库根目录 image_generation（兼容旧布局）
  """
  script_path = Path(__file__).resolve()
  candidates: List[Path] = []

  if explicit_dir:
    candidates.append(Path(explicit_dir).expanduser().resolve())

  env_dir = os.environ.get("IMG_GEN_DIR", "")
  if env_dir:
    p = Path(env_dir).expanduser().resolve()
    if p not in candidates:
      candidates.append(p)

  bundled = script_path.parents[1] / "image_generation"
  legacy = script_path.parents[2] / "image_generation"
  for p in (bundled, legacy):
    rp = p.resolve()
    if rp not in candidates:
      candidates.append(rp)

  required = [
    ("utils.py",),
    ("data", "base_scene_v5.blend"),
    ("data", "shapes_v5"),
    ("data", "materials_v5"),
    ("data", "properties.json"),
  ]

  for c in candidates:
    ok = True
    for segs in required:
      if not (c.joinpath(*segs)).exists():
        ok = False
        break
    if ok:
      return c

  tried = "\n".join([f"  - {x}" for x in candidates]) if candidates else "  (none)"
  raise FileNotFoundError(
    "Cannot locate image_generation assets. Tried:\n"
    f"{tried}\n"
    "Provide --img-gen-dir or set IMG_GEN_DIR."
  )


def place_objects(n_objects, rng, placement_range=3.0,
                  min_dist=0.25, max_retries=200):
  """CLEVR 风格放置物体，返回物体列表。"""
  if n_objects <= 6:
    placement_range = 3.0
  elif n_objects <= 8:
    placement_range = 3.5
  else:
    placement_range = 4.0

  margin = 0.40
  objects = []
  occupied = []

  for i in range(n_objects):
    shape = SHAPES[rng.integers(len(SHAPES))]
    color = COLORS[rng.integers(len(COLORS))]
    material = MATERIALS[rng.integers(len(MATERIALS))]
    size_name = rng.choice(list(SIZES.keys()))
    size_val = SIZES[size_name]

    placed = False
    for _ in range(max_retries):
      x = float(rng.uniform(-placement_range, placement_range))
      y = float(rng.uniform(-placement_range, placement_range))
      if abs(x) > placement_range - margin:
        continue
      if abs(y) > placement_range - margin:
        continue
      ok = True
      for ox, oy, os_val in occupied:
        dist = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
        if dist < min_dist + size_val + os_val:
          ok = False
          break
      if ok:
        occupied.append((x, y, size_val))
        objects.append({
          "id": f"obj_{i}",
          "shape": shape,
          "color": color,
          "material": material,
          "size": size_name,
          "position_3d": [round(x, 4), round(y, 4), round(size_val, 4)],
          "rotation": round(float(rng.uniform(0, 360)), 2),
        })
        placed = True
        break

    if not placed:
      x = float(rng.uniform(-placement_range, placement_range))
      y = float(rng.uniform(-placement_range, placement_range))
      occupied.append((x, y, size_val))
      objects.append({
        "id": f"obj_{i}",
        "shape": shape,
        "color": color,
        "material": material,
        "size": size_name,
        "position_3d": [round(x, 4), round(y, 4), round(size_val, 4)],
        "rotation": round(float(rng.uniform(0, 360)), 2),
      })

  return objects


def extract_constraints(objects, tau):
  """提取 QRR + TRR 约束。"""
  scene_dict = {
    "scene_id": "tmp",
    "objects": [
      {**o, "position_2d": [0, 0], "depth": 0.0}
      for o in objects
    ],
    "views": [{
      "camera": {
        "camera_id": "view_0",
        "position": [8.485, 8.485, 6.0],
        "look_at": [0, 0, 0],
        "fov": 50,
      }
    }],
  }

  extractor = ConstraintExtractor(ExtractionConfig(
    tau=tau,
    disjoint_pairs_only=True,
    include_qrr=True,
    include_trr=True,
    metrics=[MetricType.DIST_3D],
  ))

  osd = extractor.extract(scene_dict)

  qrr = [c.model_dump() for c in osd.world.qrr]
  trr = (
    [c.model_dump() for c in osd.views[0].trr]
    if osd.views else []
  )
  return {"qrr": qrr, "trr": trr}


def generate_all_scenes(min_obj, max_obj, per_count, tau, seed):
  """生成全部场景 JSON。"""
  rng = np.random.default_rng(seed)
  scenes = []

  for n_obj in range(min_obj, max_obj + 1):
    logger.info(f"Generating {per_count} scenes with {n_obj} objects...")
    for idx in range(per_count):
      scene_id = f"n{n_obj:02d}_{idx:03d}"
      objects = place_objects(n_obj, rng)
      constraints = extract_constraints(objects, tau)
      scenes.append({
        "scene_id": scene_id,
        "n_objects": n_obj,
        "tau": tau,
        "objects": objects,
        "constraints": constraints,
        "stats": {
          "n_qrr": len(constraints["qrr"]),
          "n_trr": len(constraints["trr"]),
        },
      })

  return scenes


# ============================================================
# Blender 渲染脚本（写入临时文件，由 Blender 执行）
# ============================================================

BLENDER_RENDER_SCRIPT = r'''
"""Blender rendering script - deterministic object placement from JSON."""
import bpy
import bpy_extras
import json
import math
import os
import sys
from mathutils import Vector

# 添加 image_generation 到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_GEN_DIR = os.environ.get("IMG_GEN_DIR", "")
if IMG_GEN_DIR and IMG_GEN_DIR not in sys.path:
    sys.path.insert(0, IMG_GEN_DIR)

import utils

def load_materials_v5(material_dir):
    """macOS Blender 5.0 兼容的材质加载（directory+filename 分开）。"""
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'):
            continue
        name = os.path.splitext(fn)[0]
        directory = os.path.join(material_dir, fn, 'NodeTree') + '/'
        bpy.ops.wm.append(directory=directory, filename=name)

def _pick_new_appended_object(before_names):
    """从 append 后的新对象中挑选刚导入的 mesh 对象。"""
    after_names = set(bpy.data.objects.keys())
    new_names = sorted(after_names - before_names)
    if not new_names:
        raise RuntimeError("append failed: no new object found")

    # 优先 mesh（形状资产应为 mesh）；若无 mesh 则退回首个新对象
    mesh_objs = [bpy.data.objects[n] for n in new_names if bpy.data.objects[n].type == 'MESH']
    if mesh_objs:
        return mesh_objs[0]
    return bpy.data.objects[new_names[0]]


def load_shape_v5(shape_dir, name, obj_name, scale, loc, theta=0):
    """
    macOS Blender 5.0 兼容的形状加载。

    关键修复:
    1) 不再依赖 bpy.context.object / 同名对象重命名猜测；
    2) 不使用 transform operator（会受 selection/context 影响）；
    3) 显式返回新导入对象并按 obj_name 绑定，避免实例串号。
    """
    blend_file = os.path.join(shape_dir, f'{name}.blend')
    directory = os.path.join(blend_file, 'Object') + '/'

    before_names = set(bpy.data.objects.keys())
    bpy.ops.wm.append(directory=directory, filename=name)
    obj = _pick_new_appended_object(before_names)

    # 确保对象名唯一且可追踪
    target_name = obj_name
    if target_name in bpy.data.objects and bpy.data.objects[target_name] != obj:
        i = 1
        while f"{target_name}_{i}" in bpy.data.objects:
            i += 1
        target_name = f"{target_name}_{i}"
    obj.name = target_name

    # 避免 selection 泄漏影响，全部显式设置
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = (0.0, 0.0, float(theta))
    obj.scale = (float(scale), float(scale), float(scale))
    obj.location = (float(loc[0]), float(loc[1]), 0.0)
    bpy.context.view_layer.update()
    return obj


def _same_object(hit_obj, target_obj):
    """判断 ray_cast 命中的对象是否与目标对象一致（兼容 evaluated/original）。"""
    if hit_obj is None or target_obj is None:
        return False
    if hit_obj == target_obj:
        return True
    if hasattr(hit_obj, "original") and (hit_obj.original == target_obj):
        return True
    return str(getattr(hit_obj, "name", "")) == str(getattr(target_obj, "name", ""))


def _world_bbox_corners(obj):
    """对象包围盒 8 角点（世界坐标）。"""
    corners = []
    for c in obj.bound_box:
        cw = obj.matrix_world @ Vector((float(c[0]), float(c[1]), float(c[2])))
        corners.append(cw)
    return corners


def _unique_points(tag_points, tol=1e-5):
    """按坐标去重，保留先出现点。"""
    out = []
    kept = []
    for tag, p in tag_points:
        keep = True
        for kp in kept:
            if (p - kp).length <= tol:
                keep = False
                break
        if keep:
            out.append((tag, p))
            kept.append(p)
    return out


def _extract_object_landmarks(obj):
    """
    提取对象真实几何 landmarks（世界坐标）。

    点集构成:
    - center: 包围盒中心
    - bbox_0..bbox_7: 包围盒角点
    - ext_{x/y/z}_{plus/minus}: 网格顶点在六个方向上的极值点
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    bbox = _world_bbox_corners(obj)
    center = Vector((0.0, 0.0, 0.0))
    if bbox:
        for p in bbox:
            center += p
        center /= float(len(bbox))
    out = [("center", center)]
    for i, p in enumerate(bbox):
        out.append((f"bbox_{i}", p))

    # 六方向极值点（来自真实 mesh 顶点）
    if mesh is not None and len(mesh.vertices) > 0:
        wverts = [eval_obj.matrix_world @ v.co for v in mesh.vertices]
        dirs = [
            ("ext_x_plus", Vector((+1.0, 0.0, 0.0))),
            ("ext_x_minus", Vector((-1.0, 0.0, 0.0))),
            ("ext_y_plus", Vector((0.0, +1.0, 0.0))),
            ("ext_y_minus", Vector((0.0, -1.0, 0.0))),
            ("ext_z_plus", Vector((0.0, 0.0, +1.0))),
            ("ext_z_minus", Vector((0.0, 0.0, -1.0))),
        ]
        for tag, d in dirs:
            best = None
            best_val = -1e30
            for p in wverts:
                val = float(p.dot(d))
                if val > best_val:
                    best_val = val
                    best = p
            if best is not None:
                out.append((tag, best))

    if mesh is not None:
        eval_obj.to_mesh_clear()
    return _unique_points(out, tol=1e-5)


def _point_visible_from_camera(scene, depsgraph, camera, target_obj, point_world):
    """
    判定 3D 点是否被当前相机可见（射线首交 + 近点一致）。

    注意:
    - 对于被遮挡点/背面点，通常返回 False
    - 对于中心点（实体内部点），通常返回 False（这是预期）
    """
    cam_pos = camera.matrix_world.translation.copy()
    vec = point_world - cam_pos
    dist = float(vec.length)
    if dist <= 1e-9:
        return False
    direction = vec.normalized()
    hit, loc, _normal, _face_idx, hit_obj, _mat = scene.ray_cast(
        depsgraph, cam_pos, direction, distance=dist + 1e-4
    )
    if not hit:
        return True
    if not _same_object(hit_obj, target_obj):
        return False
    # 命中了同一对象，但若命中点离目标 landmark 过远，视为不可见
    eps = max(1e-3, 0.01 * dist)
    return float((loc - point_world).length) <= eps


def _project_landmarks(scene, depsgraph, camera, target_obj, landmarks_world):
    """将 landmarks 投影到像素，并附带可见性标记。"""
    out = []
    for tag, pw in landmarks_world:
        px, py, pz = utils.get_camera_coords(camera, pw)
        vis = _point_visible_from_camera(scene, depsgraph, camera, target_obj, pw)
        out.append({
            "tag": str(tag),
            "pixel_coords": [int(px), int(py), float(pz)],
            "visible": bool(vis),
        })
    return out

def extract_args():
    argv = sys.argv
    if '--' in argv:
        return argv[argv.index('--') + 1:]
    return []

def main():
    args = extract_args()
    scenes_json = args[0]
    output_dir = args[1]
    base_scene = args[2]
    shape_dir = args[3]
    material_dir = args[4]
    properties_json = args[5]
    width = int(args[6])
    height = int(args[7])
    samples = int(args[8])

    with open(scenes_json, 'r') as f:
        scenes = json.load(f)

    with open(properties_json, 'r') as f:
        properties = json.load(f)

    # 颜色映射
    color_to_rgba = {}
    for name, rgb in properties['colors'].items():
        color_to_rgba[name] = [c / 255.0 for c in rgb] + [1.0]

    # 形状映射: "cube" -> "SmoothCube_v2"
    shape_to_blend = properties['shapes']

    # 材质映射: "rubber" -> "Rubber"
    mat_to_blend = properties['materials']

    # 相机配置: 4 个视角
    cameras = []
    for i in range(4):
        azimuth = 45.0 + i * 90.0
        cameras.append({
            "id": f"view_{i}",
            "azimuth": azimuth,
            "elevation": 30.0,
            "distance": 12.0,
        })

    total = len(scenes)
    for si, scene in enumerate(scenes):
        scene_id = scene["scene_id"]
        print(f"\n[{si+1}/{total}] Rendering {scene_id} "
              f"({scene['n_objects']} objects)...")

        # 创建输出目录
        mv_dir = os.path.join(output_dir, "images", "multi_view", scene_id)
        sv_dir = os.path.join(output_dir, "images", "single_view")
        meta_dir = os.path.join(output_dir, "metadata")
        os.makedirs(mv_dir, exist_ok=True)
        os.makedirs(sv_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

        # 加载基础场景
        bpy.ops.wm.open_mainfile(filepath=base_scene)
        load_materials_v5(material_dir)

        # 渲染设置
        render = bpy.context.scene.render
        render.engine = "CYCLES"
        render.resolution_x = width
        render.resolution_y = height
        render.resolution_percentage = 100
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.blur_glossy = 2.0
        # 渲染设备：默认尝试 Metal；可通过 FORCE_CPU_RENDER=1 强制 CPU
        force_cpu = os.environ.get("FORCE_CPU_RENDER", "0") == "1"
        if force_cpu:
            bpy.context.scene.cycles.device = 'CPU'
            print("  Using CPU (forced by FORCE_CPU_RENDER=1)")
        else:
            try:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = 'METAL'
                for dev in prefs.devices:
                    dev.use = True
                bpy.context.scene.cycles.device = 'GPU'
                print("  Using Metal GPU")
            except Exception:
                bpy.context.scene.cycles.device = 'CPU'
                print("  Using CPU")

        # 确定性放置物体
        objects_3d = []
        object_refs = {}
        object_geom = {}
        for obj_spec in scene["objects"]:
            shape_name = obj_spec["shape"]
            blend_name = shape_to_blend.get(shape_name, shape_name)
            size_scale = properties["sizes"][obj_spec["size"]]
            rgba = color_to_rgba[obj_spec["color"]]
            mat_blend = mat_to_blend[obj_spec["material"]]
            x, y, z = obj_spec["position_3d"]
            theta = math.radians(obj_spec["rotation"])

            # cube 需要缩小 sqrt(2)
            if blend_name in ("SmoothCube_v2", "Cube"):
                size_scale /= math.sqrt(2)

            blender_obj_name = f"{scene_id}_{obj_spec['id']}"
            obj = load_shape_v5(
                shape_dir=shape_dir,
                name=blend_name,
                obj_name=blender_obj_name,
                scale=size_scale,
                loc=(x, y),
                theta=theta,
            )
            # 创建材质并赋给物体
            mat_count = len(bpy.data.materials)
            mat = bpy.data.materials.new(name=f'Material_{mat_count}')
            mat.use_nodes = True
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            # 查找 output 节点
            output_node = None
            for n in mat.node_tree.nodes:
                if n.type == 'OUTPUT_MATERIAL':
                    output_node = n
                    break
            group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
            group_node.node_tree = bpy.data.node_groups[mat_blend]
            for inp in group_node.inputs:
                if inp.name == 'Color':
                    inp.default_value = rgba
            mat.node_tree.links.new(
                group_node.outputs['Shader'],
                output_node.inputs['Surface'],
            )
            object_refs[obj_spec["id"]] = obj

            # 提取真实几何 landmarks（世界坐标）
            lmk_world = _extract_object_landmarks(obj)
            lmk_center = None
            for tag, pw in lmk_world:
                if tag == "center":
                    lmk_center = pw
                    break
            if lmk_center is None:
                lmk_center = obj.matrix_world.translation.copy()
            origin = obj.matrix_world.translation.copy()
            object_geom[obj_spec["id"]] = {
                "center_world": lmk_center,
                "origin_world": origin,
                "landmarks_world": lmk_world,
            }
            lm3d = []
            for tag, pw in lmk_world:
                lm3d.append({
                    "tag": str(tag),
                    "coords_3d": [float(pw.x), float(pw.y), float(pw.z)],
                })

            objects_3d.append({
                "id": obj_spec["id"],
                "shape": shape_name,
                "size": obj_spec["size"],
                "material": obj_spec["material"],
                "color": obj_spec["color"],
                # 使用几何中心点（而非 origin）作为对象中心
                "3d_coords": [float(lmk_center.x), float(lmk_center.y), float(lmk_center.z)],
                "origin_3d": [float(origin.x), float(origin.y), float(origin.z)],
                "rotation": obj_spec["rotation"],
                "landmarks_3d": lm3d,
                "landmark_source": "mesh_extrema_bbox",
            })

        # 渲染多视角
        views_meta = []
        for cam_cfg in cameras:
            az = math.radians(cam_cfg["azimuth"])
            el = math.radians(cam_cfg["elevation"])
            d = cam_cfg["distance"]

            cx = d * math.cos(el) * math.cos(az)
            cy = d * math.cos(el) * math.sin(az)
            cz = d * math.sin(el)

            camera = bpy.data.objects['Camera']
            camera.location = (cx, cy, cz)
            direction = Vector((0, 0, 0)) - Vector((cx, cy, cz))
            rot_quat = direction.to_track_quat('-Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()

            # 先渲染，确保相机状态与图像一致，再写像素坐标
            img_path = os.path.join(mv_dir, f"{cam_cfg['id']}.png")
            render.filepath = img_path
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()

            # 计算像素坐标（基于实际 Blender 对象位置）
            objs_with_px = []
            for o3d in objects_3d:
                oid = o3d["id"]
                obj_ref = object_refs.get(o3d["id"])
                geom = object_geom.get(oid, {})
                if obj_ref is not None:
                    pos = geom.get("center_world", obj_ref.matrix_world.translation.copy())
                    lmk_world = geom.get("landmarks_world", [])
                else:
                    pos = Vector(o3d["3d_coords"])
                    lmk_world = []
                px, py, pz = utils.get_camera_coords(camera, pos)
                o_copy = dict(o3d)
                o_copy["pixel_coords"] = [px, py, float(pz)]
                o_copy["landmarks_2d"] = _project_landmarks(
                    scene=bpy.context.scene,
                    depsgraph=depsgraph,
                    camera=camera,
                    target_obj=obj_ref,
                    landmarks_world=lmk_world,
                )
                objs_with_px.append(o_copy)

            views_meta.append({
                "view_id": cam_cfg["id"],
                "image_path": f"{cam_cfg['id']}.png",
                "camera": {
                    "azimuth": cam_cfg["azimuth"],
                    "elevation": cam_cfg["elevation"],
                    "distance": cam_cfg["distance"],
                    "position": [cx, cy, cz],
                },
                "objects": objs_with_px,
            })

        # 复制 view_0 作为单视角图
        import shutil
        v0_src = os.path.join(mv_dir, "view_0.png")
        v0_dst = os.path.join(sv_dir, f"{scene_id}.png")
        if os.path.exists(v0_src):
            shutil.copy2(v0_src, v0_dst)

        # 保存元数据（含约束）
        metadata = {
            "scene_id": scene_id,
            "n_objects": scene["n_objects"],
            "tau": scene["tau"],
            "objects": objects_3d,
            "constraints": scene["constraints"],
            "views": views_meta,
        }
        meta_path = os.path.join(meta_dir, f"{scene_id}.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Done: {len(views_meta)} views rendered")

    print(f"\nAll {total} scenes rendered successfully!")


main()
'''


def run_blender_render(scenes, output_dir, blender_path, img_gen_dir,
                       width, height, samples):
  """调用 Blender headless 渲染所有场景。"""
  base_scene = os.path.join(img_gen_dir, "data", "base_scene_v5.blend")
  shape_dir = os.path.join(img_gen_dir, "data", "shapes_v5")
  material_dir = os.path.join(img_gen_dir, "data", "materials_v5")
  properties_json = os.path.join(img_gen_dir, "data", "properties.json")

  # 检查资源文件
  for p in [base_scene, shape_dir, material_dir, properties_json]:
    if not os.path.exists(p):
      raise FileNotFoundError(f"Missing: {p}")

  # 写入场景 JSON 到临时文件
  scenes_tmp = os.path.join(output_dir, "_scenes_for_render.json")
  with open(scenes_tmp, 'w') as f:
    json.dump(scenes, f)

  # 写入 Blender 脚本到临时文件
  script_tmp = os.path.join(output_dir, "_render_script.py")
  with open(script_tmp, 'w') as f:
    f.write(BLENDER_RENDER_SCRIPT)

  # 构造命令
  cmd = [
    blender_path,
    "--background",
    "--python", script_tmp,
    "--",
    scenes_tmp,
    output_dir,
    base_scene,
    shape_dir,
    material_dir,
    properties_json,
    str(width),
    str(height),
    str(samples),
  ]

  env = os.environ.copy()
  env["IMG_GEN_DIR"] = img_gen_dir

  logger.info(f"Starting Blender rendering ({len(scenes)} scenes)...")
  logger.info(f"  Blender: {blender_path}")
  logger.info(f"  Resolution: {width}x{height}, samples: {samples}")

  proc = subprocess.run(
    cmd, env=env,
    capture_output=False,
    timeout=3600,  # 1 小时超时
  )

  if proc.returncode != 0:
    raise RuntimeError(f"Blender exited with code {proc.returncode}")

  # 清理临时文件
  os.remove(scenes_tmp)
  os.remove(script_tmp)


def save_summary(scenes, output_dir, config):
  """保存数据集摘要和索引文件。"""
  # 按物体数量统计
  groups = {}
  for s in scenes:
    n = s["n_objects"]
    if n not in groups:
      groups[n] = []
    groups[n].append(s)

  summary = {
    "config": config,
    "total_scenes": len(scenes),
    "total_images": len(scenes) * 5,  # 1 single + 4 multi
    "groups": {},
  }

  for n in sorted(groups.keys()):
    g = groups[n]
    summary["groups"][f"{n}_objects"] = {
      "n_scenes": len(g),
      "n_objects": n,
      "n_qrr_per_scene": g[0]["stats"]["n_qrr"],
      "n_trr_per_scene": g[0]["stats"]["n_trr"],
    }

  with open(os.path.join(output_dir, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

  # 场景索引文件
  index = []
  for s in scenes:
    sid = s["scene_id"]
    index.append({
      "scene_id": sid,
      "n_objects": s["n_objects"],
      "single_view": f"images/single_view/{sid}.png",
      "multi_view": [
        f"images/multi_view/{sid}/view_{i}.png"
        for i in range(4)
      ],
      "metadata": f"metadata/{sid}.json",
    })

  with open(os.path.join(output_dir, "index.json"), 'w') as f:
    json.dump(index, f, indent=2, ensure_ascii=False)

  # 按物体数量的分组索引
  by_count_dir = os.path.join(output_dir, "by_count")
  os.makedirs(by_count_dir, exist_ok=True)
  for n in sorted(groups.keys()):
    group_index = [
      e for e in index if e["n_objects"] == n
    ]
    with open(os.path.join(by_count_dir, f"{n}_objects.json"), 'w') as f:
      json.dump(group_index, f, indent=2, ensure_ascii=False)


def main():
  parser = argparse.ArgumentParser(
    description="Generate test scenes and render with Blender"
  )
  parser.add_argument(
    "--output-dir", "-o", default="./data",
    help="Output directory (default: ./data)",
  )
  parser.add_argument(
    "--blender-path", "-b",
    default="/Applications/Blender.app/Contents/MacOS/Blender",
    help="Blender executable path",
  )
  parser.add_argument(
    "--min-objects", type=int, default=3,
  )
  parser.add_argument(
    "--max-objects", type=int, default=10,
  )
  parser.add_argument(
    "--scenes-per-count", "-n", type=int, default=10,
  )
  parser.add_argument(
    "--tau", type=float, default=0.10,
  )
  parser.add_argument(
    "--seed", type=int, default=42,
  )
  parser.add_argument(
    "--width", type=int, default=480,
  )
  parser.add_argument(
    "--height", type=int, default=320,
  )
  parser.add_argument(
    "--samples", type=int, default=128,
    help="Cycles render samples (default: 128, lower=faster)",
  )
  parser.add_argument(
    "--img-gen-dir", type=str, default="",
    help="image_generation 目录（可选，默认自动解析）",
  )

  args = parser.parse_args()

  # 路径
  img_gen_dir = str(resolve_image_generation_dir(args.img_gen_dir))
  output_dir = os.path.abspath(args.output_dir)

  total = (args.max_objects - args.min_objects + 1) * args.scenes_per_count

  logger.info("=" * 60)
  logger.info("Scene Generation + Blender Rendering")
  logger.info(f"  Objects: {args.min_objects}-{args.max_objects}")
  logger.info(f"  Scenes per count: {args.scenes_per_count}")
  logger.info(f"  Total scenes: {total}")
  logger.info(f"  Images: {total * 5} (1 single + 4 multi per scene)")
  logger.info(f"  Resolution: {args.width}x{args.height}")
  logger.info(f"  Samples: {args.samples}")
  logger.info(f"  image_generation: {img_gen_dir}")
  logger.info(f"  Output: {output_dir}")
  logger.info("=" * 60)

  # 第一步：生成场景数据
  logger.info("\n[Step 1/3] Generating scene data...")
  scenes = generate_all_scenes(
    args.min_objects, args.max_objects,
    args.scenes_per_count, args.tau, args.seed,
  )
  logger.info(f"  Generated {len(scenes)} scenes")

  # 创建输出目录
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(os.path.join(output_dir, "images", "single_view"), exist_ok=True)
  os.makedirs(os.path.join(output_dir, "images", "multi_view"), exist_ok=True)
  os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

  # 第二步：Blender 渲染
  logger.info("\n[Step 2/3] Rendering with Blender (headless)...")
  run_blender_render(
    scenes, output_dir, args.blender_path, img_gen_dir,
    args.width, args.height, args.samples,
  )

  # 第三步：保存索引和摘要
  logger.info("\n[Step 3/3] Saving index and summary...")
  config = {
    "min_objects": args.min_objects,
    "max_objects": args.max_objects,
    "scenes_per_count": args.scenes_per_count,
    "tau": args.tau,
    "seed": args.seed,
    "width": args.width,
    "height": args.height,
    "samples": args.samples,
  }
  save_summary(scenes, output_dir, config)

  logger.info("\n" + "=" * 60)
  logger.info("Done!")
  logger.info(f"  {output_dir}/")
  logger.info(f"    images/single_view/  - {total} images")
  logger.info(f"    images/multi_view/   - {total}x4 images")
  logger.info(f"    metadata/            - {total} JSON files")
  logger.info(f"    by_count/            - grouped indices")
  logger.info(f"    index.json           - full index")
  logger.info(f"    summary.json         - statistics")
  logger.info("=" * 60)


if __name__ == "__main__":
  main()
