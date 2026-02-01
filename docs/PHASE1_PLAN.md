# Phase 1: 数据基础设施 - 详细计划

> 目标：构建稳定的 Benchmark 数据集生成流水线
> 预计工作量：3-5 天

---

## 一、当前状态分析

### 1.1 已完成 ✅

| 组件 | 文件 | 状态 |
|------|------|------|
| 单视角渲染 | `image_generation/render_images.py` | ✅ 完整可用 |
| 基础场景模板 | `image_generation/data/base_scene_v5.blend` | ✅ Blender 5.0 兼容 |
| DSL 多视角结构 | `ordinal_spatial/dsl/schema.py` | ✅ CameraParams, ViewConstraints 已定义 |
| 多视角约束提取 | `ordinal_spatial/generation/constraint_extractor.py` | ✅ 支持多视角 |
| VLM Agent 多视角接口 | `ordinal_spatial/agents/vlm_constraint_agent.py` | ✅ extract_from_multi_view() |
| 场景 JSON 生成 | `ordinal_spatial/scripts/generate_dataset.py` | ✅ 生成约束数据 |

### 1.2 缺失 ❌

| 组件 | 说明 |
|------|------|
| **多视角渲染脚本** | render_images.py 只支持单相机，无法生成同一场景的多视角图像 |
| **相机轨道配置** | 当前相机只有 jitter，无法系统性地生成多视角 |
| **渲染与约束的集成** | generate_dataset.py 只生成 JSON，不调用 Blender 渲染 |
| **数据集验证工具** | 无图像-约束一致性检查 |
| **批量渲染脚本** | 无端到端的数据集构建流程 |

---

## 二、详细任务分解

### Task 1.1: 多视角渲染脚本

**目标**: 创建 `image_generation/render_multiview.py`

**输入**:
- 场景配置（物体列表、位置、属性）
- 相机轨道参数

**输出**:
- N 张不同视角的渲染图像
- 每个视角的相机参数 JSON
- 更新后的场景元数据

#### 1.1.1 相机轨道设计

```
俯视图 (Top View):

        View 1 (0°)
            ●
            |
            |
   View 4 --+-- View 2
   (270°)   |   (90°)
            |
            ●
        View 3 (180°)

侧视图 (Side View):

    Camera ●-------- elevation angle (30°)
            \
             \
              \
               ● Scene Center (0,0,0)
```

**默认参数** (可配置):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_views` | 4 | 视角数量 |
| `camera_distance` | 12.0 | 相机到场景中心的距离 |
| `elevation` | 30° | 仰角（固定） |
| `azimuth_start` | 0° | 起始方位角 |
| `azimuth_step` | 90° | 方位角步长 (360° / n_views) |

#### 1.1.2 核心函数设计

```python
# image_generation/render_multiview.py

@dataclass
class CameraConfig:
    """单个相机的配置"""
    camera_id: str
    azimuth: float      # 方位角 (degrees)
    elevation: float    # 仰角 (degrees)
    distance: float     # 到场景中心的距离
    look_at: Tuple[float, float, float] = (0, 0, 0)

@dataclass
class MultiViewConfig:
    """多视角渲染配置"""
    n_views: int = 4
    camera_distance: float = 12.0
    elevation: float = 30.0
    azimuth_start: float = 0.0
    resolution: Tuple[int, int] = (480, 320)

    def generate_cameras(self) -> List[CameraConfig]:
        """生成相机配置列表"""
        cameras = []
        azimuth_step = 360.0 / self.n_views
        for i in range(self.n_views):
            azimuth = self.azimuth_start + i * azimuth_step
            cameras.append(CameraConfig(
                camera_id=f"view_{i}",
                azimuth=azimuth,
                elevation=self.elevation,
                distance=self.camera_distance
            ))
        return cameras

def render_multiview_scene(
    scene_config: dict,
    output_dir: str,
    mv_config: MultiViewConfig,
    args: argparse.Namespace
) -> dict:
    """
    渲染单个场景的多视角图像

    Returns:
        scene_metadata: 包含所有视角信息的场景元数据
    """
    ...

def position_camera(camera_config: CameraConfig) -> None:
    """
    根据配置设置 Blender 相机位置

    球坐标转笛卡尔坐标:
    x = distance * cos(elevation) * cos(azimuth)
    y = distance * cos(elevation) * sin(azimuth)
    z = distance * sin(elevation)
    """
    ...
```

#### 1.1.3 输出结构

```
output/
├── scenes/
│   └── scene_000000/
│       ├── metadata.json        # 场景元数据（物体、约束）
│       ├── view_0.png           # 视角 0 图像
│       ├── view_0_camera.json   # 视角 0 相机参数
│       ├── view_1.png
│       ├── view_1_camera.json
│       ├── view_2.png
│       ├── view_2_camera.json
│       ├── view_3.png
│       └── view_3_camera.json
```

**metadata.json 结构**:
```json
{
  "scene_id": "scene_000000",
  "objects": [...],
  "world_constraints": {
    "qrr_3d": [...],
    "axial_3d": [...],
    "topology": [...],
    "size": [...]
  },
  "views": [
    {
      "view_id": "view_0",
      "image_path": "view_0.png",
      "camera": {
        "azimuth": 0,
        "elevation": 30,
        "distance": 12,
        "position": [10.39, 0, 6],
        "look_at": [0, 0, 0]
      },
      "view_constraints": {
        "qrr_2d": [...],
        "occlusion": [...]
      }
    },
    ...
  ]
}
```

---

### Task 1.2: Benchmark 数据集生成

**目标**: 创建 `ordinal_spatial/scripts/build_benchmark.py`

#### 1.2.1 数据集规模

| Split | 场景数 | 单视角图像 | 多视角图像 (4x) | 用途 |
|-------|--------|-----------|----------------|------|
| train | 1000 | 1000 | 4000 | 训练（如需要） |
| val | 200 | 200 | 800 | 验证/调参 |
| test_iid | 200 | 200 | 800 | IID 测试 |
| test_comp | 100 | 100 | 400 | 组合泛化（更多物体） |
| test_hard | 100 | 100 | 400 | 困难（小 tau） |
| **Total** | **1600** | **1600** | **6400** | |

#### 1.2.2 场景配置差异

| Split | min_objects | max_objects | tau | 说明 |
|-------|-------------|-------------|-----|------|
| train | 4 | 10 | 0.10 | 标准配置 |
| val | 4 | 10 | 0.10 | 同 train |
| test_iid | 4 | 10 | 0.10 | 同 train |
| test_comp | 10 | 15 | 0.10 | 更多物体，测试组合泛化 |
| test_hard | 4 | 10 | 0.05 | 更小容差，更难区分 |

#### 1.2.3 构建流程

```
┌─────────────────────────────────────────────────────────┐
│                 build_benchmark.py                       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: 生成场景配置                                    │
│  - 随机物体位置、大小、颜色                              │
│  - 检查有效性（无重叠、无退化）                          │
│  - 输出: scene_configs.json                             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: 调用 Blender 渲染                              │
│  - 加载场景配置                                          │
│  - 多视角渲染                                            │
│  - 输出: images/ + camera JSONs                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: 提取 Ground Truth 约束                         │
│  - 调用 BlenderConstraintAgent                          │
│  - 计算 3D 约束 + 每视角 2D 约束                         │
│  - 输出: constraints.json                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: 组装最终数据集                                  │
│  - 合并元数据                                            │
│  - 划分 train/val/test                                  │
│  - 输出: splits/*.json                                  │
└─────────────────────────────────────────────────────────┘
```

#### 1.2.4 最终目录结构

```
data/ordinal_spatial_benchmark/
├── images/
│   ├── single_view/
│   │   ├── train_000000.png
│   │   ├── train_000001.png
│   │   └── ...
│   └── multi_view/
│       ├── train_000000/
│       │   ├── view_0.png
│       │   ├── view_1.png
│       │   ├── view_2.png
│       │   └── view_3.png
│       └── ...
├── splits/
│   ├── train.json
│   ├── val.json
│   ├── test_iid.json
│   ├── test_comp.json
│   └── test_hard.json
├── metadata/
│   ├── scene_000000.json
│   └── ...
└── dataset_info.json
```

**splits/*.json 格式**:
```json
[
  {
    "scene_id": "train_000000",
    "single_view_image": "images/single_view/train_000000.png",
    "multi_view_images": [
      "images/multi_view/train_000000/view_0.png",
      "images/multi_view/train_000000/view_1.png",
      "images/multi_view/train_000000/view_2.png",
      "images/multi_view/train_000000/view_3.png"
    ],
    "metadata_path": "metadata/train_000000.json",
    "n_objects": 6,
    "tau": 0.10
  },
  ...
]
```

---

### Task 1.3: 数据集验证

**目标**: 创建 `ordinal_spatial/scripts/validate_benchmark.py`

#### 1.3.1 验证项

| 检查项 | 方法 | 严重程度 |
|--------|------|----------|
| 图像完整性 | 文件存在 + 可读取 | CRITICAL |
| 图像尺寸一致 | PIL.Image.size | CRITICAL |
| JSON 格式正确 | json.load() 无异常 | CRITICAL |
| 物体数量匹配 | len(objects) == 实际渲染 | CRITICAL |
| 约束非空 | len(qrr) > 0 | WARNING |
| 约束一致性 | 无循环矛盾 | WARNING |
| 视角覆盖 | 4 个视角都存在 | CRITICAL |

#### 1.3.2 统计报告

```
================== Benchmark Validation Report ==================

Dataset: data/ordinal_spatial_benchmark/
Generated: 2026-02-01 15:30:00

Split Statistics:
┌──────────┬────────┬────────────┬─────────────┬──────────────┐
│ Split    │ Scenes │ Avg Objects│ Avg QRR     │ Avg TRR      │
├──────────┼────────┼────────────┼─────────────┼──────────────┤
│ train    │ 1000   │ 7.2        │ 156.3       │ 84.1         │
│ val      │ 200    │ 7.1        │ 152.8       │ 82.4         │
│ test_iid │ 200    │ 7.3        │ 158.9       │ 85.7         │
│ test_comp│ 100    │ 12.4       │ 412.6       │ 198.3        │
│ test_hard│ 100    │ 7.0        │ 148.2       │ 81.9         │
└──────────┴────────┴────────────┴─────────────┴──────────────┘

Image Statistics:
- Total single-view images: 1600
- Total multi-view images: 6400
- Resolution: 480x320
- Format: PNG

Validation Results:
✅ All images exist and readable
✅ All JSON files valid
✅ Object counts match
✅ No constraint inconsistencies
⚠️  3 scenes have fewer than expected QRR constraints

Errors: 0
Warnings: 3
```

---

## 三、实现顺序与依赖

```
Week 1:
├── Day 1-2: Task 1.1 多视角渲染脚本
│   ├── 相机轨道计算
│   ├── Blender 相机定位
│   └── 单场景多视角渲染测试
│
├── Day 3-4: Task 1.2 数据集生成
│   ├── 集成渲染脚本
│   ├── 批量渲染流程
│   └── 约束提取集成
│
└── Day 5: Task 1.3 验证与调试
    ├── 验证脚本
    ├── 修复问题
    └── 生成最终数据集
```

---

## 四、技术细节

### 4.1 相机位置计算

球坐标 → 笛卡尔坐标转换：

```python
import math

def spherical_to_cartesian(distance, azimuth_deg, elevation_deg):
    """
    将球坐标转换为笛卡尔坐标

    Args:
        distance: 到原点的距离
        azimuth_deg: 方位角（度），0° 为 +X 方向，逆时针
        elevation_deg: 仰角（度），0° 为水平面

    Returns:
        (x, y, z) 笛卡尔坐标
    """
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)

    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)

    return (x, y, z)

# 示例：4 视角配置
# View 0: azimuth=0°   → (10.39, 0, 6)
# View 1: azimuth=90°  → (0, 10.39, 6)
# View 2: azimuth=180° → (-10.39, 0, 6)
# View 3: azimuth=270° → (0, -10.39, 6)
```

### 4.2 Blender 相机设置

```python
import bpy
import mathutils

def set_camera_position(position, look_at=(0, 0, 0)):
    """
    设置 Blender 相机位置和朝向
    """
    camera = bpy.data.objects['Camera']

    # 设置位置
    camera.location = position

    # 计算朝向
    direction = mathutils.Vector(look_at) - mathutils.Vector(position)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
```

### 4.3 与现有 render_images.py 的关系

**方案 A**: 扩展现有脚本（推荐）
- 添加 `--multiview` 参数
- 添加 `--n_views` 参数
- 复用现有的物体放置、关系计算逻辑

**方案 B**: 创建独立脚本
- 新建 `render_multiview.py`
- 导入 `render_images.py` 的核心函数
- 添加多视角循环

**选择方案 A**，因为：
- 避免代码重复
- 保持渲染逻辑一致
- 更易维护

---

## 五、验收标准

### Task 1.1 完成标准 ✅ IMPLEMENTED

- [x] `render_multiview.py` 创建完成
- [x] 支持 N 视角渲染 (默认 4 视角)
- [x] 相机轨道配置（球坐标系）
- [x] 每个视角的相机参数正确记录
- [ ] 需要 Blender 环境测试实际渲染

### Task 1.2 完成标准 ✅ IMPLEMENTED

- [x] `build_benchmark.py` 创建完成
- [x] 支持完整的 train/val/test 划分
- [x] 集成渲染和约束提取
- [ ] 需要 Blender 环境运行端到端测试

### Task 1.3 完成标准 ✅ IMPLEMENTED

- [x] `validate_benchmark.py` 创建完成
- [x] 完整的验证检查（图像、JSON、一致性）
- [x] 统计报告生成
- [ ] 需要实际数据集测试

### 额外完成：Constraint-Diff 指标 ✅

- [x] `constraint_diff.py` 创建完成
- [x] Missing/Spurious/Violated 三元组指标
- [x] 支持多约束类型 (QRR, axial, size, topology)
- [x] 批量评估支持

---

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Blender 渲染慢 | 数据集生成时间长 | 使用 GPU 渲染；降低分辨率；并行渲染 |
| 相机遮挡问题 | 某些视角物体被遮挡 | 选择合适的仰角；检测并跳过严重遮挡场景 |
| 磁盘空间不足 | 无法存储所有图像 | 预估空间需求；使用压缩格式 |
| 约束计算错误 | GT 不准确 | 对比 Oracle baseline；人工抽查 |

**磁盘空间估算**:
- 单张图像: ~50KB (480x320 PNG)
- 单视角总计: 1600 × 50KB = 80MB
- 多视角总计: 6400 × 50KB = 320MB
- JSON 元数据: ~100MB
- **总计: ~500MB**

---

## 七、待确认问题

在实现前需要确认：

1. **视角数量**: 默认 4 视角是否足够？是否需要支持更多？
2. **分辨率**: 480×320 是否合适？VLM 是否需要更高分辨率？
3. **单视角选择**: test 时的"单视角"是否固定为 view_0，还是随机选择？
4. **相机仰角**: 30° 是否合适？是否需要变化的仰角？

---

## 八、文件清单

### 新建文件

| 文件 | 说明 |
|------|------|
| `image_generation/render_multiview.py` | 或扩展 render_images.py |
| `ordinal_spatial/scripts/build_benchmark.py` | 数据集构建主脚本 |
| `ordinal_spatial/scripts/validate_benchmark.py` | 数据集验证脚本 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `image_generation/render_images.py` | 添加多视角支持（如采用方案 A） |
| `ordinal_spatial/scripts/generate_dataset.py` | 可能需要与渲染流程集成 |
