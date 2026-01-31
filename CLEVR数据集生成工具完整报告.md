# CLEVR 数据集生成工具完整报告

## 目录

1. [项目概述](#1-项目概述)
2. [项目结构](#2-项目结构)
3. [核心模块详解](#3-核心模块详解)
4. [依赖项和版本要求](#4-依赖项和版本要求)
5. [Blender 5.0 兼容性分析](#5-blender-50-兼容性分析)
6. [API 使用指南](#6-api-使用指南)
7. [数据格式说明](#7-数据格式说明)
8. [迁移建议](#8-迁移建议)

---

## 1. 项目概述

### 1.1 项目简介

CLEVR (Compositional Language and Elementary Visual Reasoning) 数据集生成工具是由 Facebook AI Research 开发的合成数据集生成框架。该工具用于生成包含简单3D几何体的场景图像及相应的视觉推理问题。

### 1.2 论文引用

```bibtex
@inproceedings{johnson2017clevr,
  title={CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
  author={Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens
          and Fei-Fei, Li and Zitnick, C Lawrence and Girshick, Ross},
  booktitle={CVPR},
  year={2017}
}
```

### 1.3 主要功能

| 功能 | 描述 |
|------|------|
| 3D场景渲染 | 使用Blender生成随机3D几何场景 |
| 场景标注 | 自动生成物体属性、位置、空间关系的JSON标注 |
| 问题生成 | 基于模板生成自然语言问题和函数式程序 |
| 可组合推理 | 支持多步推理问题的生成 |

---

## 2. 项目结构

```
clevr-dataset-gen/
├── image_generation/           # 图像生成模块
│   ├── render_images.py        # 主渲染脚本
│   ├── utils.py                # Blender工具函数
│   ├── collect_scenes.py       # 场景JSON合并工具
│   ├── data/
│   │   ├── base_scene.blend    # 基础场景文件（相机、灯光、地面）
│   │   ├── properties.json     # 物体属性定义
│   │   ├── shapes/             # 形状模型 (.blend文件)
│   │   ├── materials/          # 材质文件 (.blend文件)
│   │   ├── CoGenT_A.json       # CLEVR-CoGenT A配置
│   │   └── CoGenT_B.json       # CLEVR-CoGenT B配置
│   └── README.md
│
├── question_generation/        # 问题生成模块
│   ├── generate_questions.py   # 主问题生成脚本
│   ├── question_engine.py      # 问题引擎核心
│   ├── metadata.json           # 函数语言元数据
│   ├── synonyms.json           # 同义词映射
│   ├── CLEVR_1.0_templates/    # 问题模板目录
│   │   ├── zero_hop.json       # 直接属性查询
│   │   ├── one_hop.json        # 单步关系查询
│   │   ├── two_hop.json        # 双步关系查询
│   │   ├── three_hop.json      # 三步关系查询
│   │   ├── comparison.json     # 属性比较
│   │   ├── compare_integer.json # 数量比较
│   │   ├── same_relate.json    # 关系一致性
│   │   ├── single_and.json     # AND逻辑
│   │   └── single_or.json      # OR逻辑
│   └── README.md
│
├── ordinal_spatial/            # 空间推理评估框架（新增）
│
├── LICENSE                     # BSD许可证
├── README.md                   # 项目文档
├── CODE_OF_CONDUCT.md          # 行为准则
├── CONTRIBUTING.md             # 贡献指南
└── PATENTS                     # 专利声明
```

---

## 3. 核心模块详解

### 3.1 图像生成模块 (image_generation/)

#### 3.1.1 工作流程

```
1. 加载基础场景 (base_scene.blend)
      ↓
2. 随机添加 3-10 个物体
      ↓
3. 为每个物体分配随机属性
   - 形状: cube / sphere / cylinder
   - 颜色: 8种预定义颜色
   - 材质: rubber / metal
   - 尺寸: large (0.7) / small (0.35)
      ↓
4. 验证物体可见性（每个物体≥200像素）
      ↓
5. 使用Cycles渲染器渲染图像
      ↓
6. 生成场景JSON文件
```

#### 3.1.2 物体属性配置 (properties.json)

```json
{
  "shapes": {
    "cube": "SmoothCube_v2",
    "sphere": "Sphere",
    "cylinder": "SmoothCylinder"
  },
  "colors": {
    "gray": [87, 87, 87],
    "red": [173, 35, 35],
    "blue": [42, 75, 215],
    "green": [29, 105, 20],
    "brown": [129, 74, 25],
    "purple": [129, 38, 192],
    "cyan": [41, 208, 208],
    "yellow": [255, 238, 51]
  },
  "materials": {
    "rubber": "Rubber",
    "metal": "MyMetal"
  },
  "sizes": {
    "large": 0.7,
    "small": 0.35
  }
}
```

#### 3.1.3 主要脚本参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--num_images` | 5 | 生成图像数量 |
| `--min_objects` | 3 | 最少物体数 |
| `--max_objects` | 10 | 最多物体数 |
| `--width` | 320 | 图像宽度(像素) |
| `--height` | 240 | 图像高度(像素) |
| `--use_gpu` | 0 | 启用GPU渲染 |
| `--render_num_samples` | 512 | 光线追踪采样数 |
| `--min_pixels_per_object` | 200 | 物体最小可见像素 |
| `--min_dist` | 0.25 | 物体最小间距 |
| `--margin` | 0.4 | 方向性边距 |

### 3.2 问题生成模块 (question_generation/)

#### 3.2.1 问题类型

| 模板文件 | 类型 | 示例 |
|----------|------|------|
| `zero_hop.json` | 直接查询 | "What color is the large cube?" |
| `one_hop.json` | 单步推理 | "What color is the object left of the sphere?" |
| `two_hop.json` | 双步推理 | "What material is the thing behind the red cube?" |
| `three_hop.json` | 三步推理 | 复杂空间关系链 |
| `comparison.json` | 属性比较 | "Is the cube the same color as the sphere?" |
| `compare_integer.json` | 数量比较 | "Are there more cubes than spheres?" |
| `same_relate.json` | 关系一致 | "Does the red thing have the same size as the blue thing?" |

#### 3.2.2 函数式程序结构

每个问题都有对应的函数式程序表示：

```json
{
  "program": [
    {"type": "scene", "inputs": []},
    {"type": "filter_color", "inputs": [0], "side_inputs": ["red"]},
    {"type": "filter_shape", "inputs": [1], "side_inputs": ["cube"]},
    {"type": "unique", "inputs": [2]},
    {"type": "query_size", "inputs": [3]}
  ]
}
```

---

## 4. 依赖项和版本要求

### 4.1 官方推荐版本

| 组件 | 版本 | 说明 |
|------|------|------|
| **Blender** | 2.78c | 官方推荐版本 |
| **Blender Python** | 3.5 | Blender 2.78c 捆绑 |
| **Python (问题生成)** | 2.7 / 3.5+ | 独立运行 |
| **操作系统** | OSX / Ubuntu 16.04 | 测试环境 |

### 4.2 Python依赖

问题生成模块仅需标准库，无额外依赖。

### 4.3 Blender Python环境配置

需要将 `image_generation` 添加到 Blender Python 的 site-packages：

```bash
# Linux 示例
echo $PWD/image_generation >> $BLENDER_PATH/2.78/python/lib/python3.5/site-packages/clevr.pth

# macOS 示例
echo $PWD/image_generation >> /Applications/blender/blender.app/Contents/Resources/2.78/python/lib/python3.5/site-packages/clevr.pth
```

---

## 5. Blender 5.0 兼容性分析

### 5.1 总体评估

> **结论：原始代码无法直接在 Blender 5.0 上运行，需要进行重大修改。**

Blender 从 2.79 到 2.80 进行了革命性的 API 重构，之后的版本(包括5.0)继承了这些变化。以下是详细的兼容性问题分析：

### 5.2 关键API变更对照表

| 问题类别 | 原代码 (2.78) | Blender 5.0 | 影响文件 |
|----------|---------------|-------------|----------|
| **对象选择** | `obj.select = True` | `obj.select_set(True)` | utils.py:40-41 |
| **活动对象** | `bpy.context.scene.objects.active = obj` | `bpy.context.view_layer.objects.active = obj` | utils.py:103 |
| **用户偏好** | `bpy.context.user_preferences` | `bpy.context.preferences` | render_images.py:239,242 |
| **渲染引擎** | `'BLENDER_RENDER'` | 已移除，使用 `'BLENDER_EEVEE'` | render_images.py:517 |
| **抗锯齿** | `render_args.use_antialiasing` | 已移除 | render_images.py:513,518,558 |
| **图层系统** | `obj.layers[idx]` | Collections系统 | utils.py:68-74 |
| **材质无影** | `mat.use_shadeless = True` | 使用Emission shader | render_images.py:539 |
| **漫反射颜色** | `mat.diffuse_color = [r,g,b]` | `mat.diffuse_color = [r,g,b,a]` (RGBA) | render_images.py:538 |
| **平面创建** | `primitive_plane_add(radius=5)` | `primitive_plane_add(size=5)` | render_images.py:264 |
| **Cycles世界设置** | `world.cycles.sample_as_light` | 已更改或移除 | render_images.py:246 |
| **瓦片渲染** | `render_args.tile_x/y` | 已移除(Cycles自动优化) | render_images.py:234-235 |

### 5.3 详细问题分析

#### 5.3.1 对象选择系统 (严重)

**位置**: `utils.py` 第37-42行

```python
# 原代码 (Blender 2.78)
def delete_object(obj):
    for o in bpy.data.objects:
        o.select = False      # ❌ 2.80+中已移除
    obj.select = True         # ❌ 2.80+中已移除
    bpy.ops.object.delete()
```

**修复方案**:
```python
# Blender 2.80+ / 5.0
def delete_object(obj):
    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.ops.object.delete()
```

#### 5.3.2 活动对象设置 (严重)

**位置**: `utils.py` 第103行

```python
# 原代码
bpy.context.scene.objects.active = bpy.data.objects[new_name]  # ❌

# 修复
bpy.context.view_layer.objects.active = bpy.data.objects[new_name]  # ✅
```

#### 5.3.3 用户偏好访问 (严重)

**位置**: `render_images.py` 第238-243行

```python
# 原代码
if bpy.app.version < (2, 78, 0):
    bpy.context.user_preferences.system.compute_device_type = 'CUDA'  # ❌
else:
    cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences  # ❌

# 修复 (Blender 2.80+)
cycles_prefs = bpy.context.preferences.addons['cycles'].preferences  # ✅
```

#### 5.3.4 渲染引擎 (严重)

**位置**: `render_images.py` 第517行

```python
# 原代码
render_args.engine = 'BLENDER_RENDER'  # ❌ 已在2.80中移除

# 修复方案
render_args.engine = 'BLENDER_EEVEE'   # ✅ 用于快速渲染
# 或
render_args.engine = 'CYCLES'          # ✅ 用于高质量渲染
```

#### 5.3.5 图层系统 → Collections (严重)

**位置**: `utils.py` 第68-74行, `render_images.py` 第520-553行

Blender 2.80完全重构了图层系统，用Collections替代：

```python
# 原代码
def set_layer(obj, layer_idx):
    obj.layers[layer_idx] = True  # ❌
    for i in range(len(obj.layers)):
        obj.layers[i] = (i == layer_idx)

# 修复思路 (使用Collections)
def move_to_collection(obj, collection_name):
    # 从所有collection中移除
    for col in obj.users_collection:
        col.objects.unlink(obj)
    # 添加到目标collection
    target_col = bpy.data.collections.get(collection_name)
    if target_col is None:
        target_col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(target_col)
    target_col.objects.link(obj)
```

#### 5.3.6 材质系统 (严重)

**位置**: `render_images.py` 第526-540行

```python
# 原代码 - 无阴影材质用于可见性检测
mat.diffuse_color = [r, g, b]    # ❌ 需要RGBA
mat.use_shadeless = True          # ❌ 已移除

# 修复 - 使用Emission shader
mat.diffuse_color = [r, g, b, 1.0]  # ✅
mat.use_nodes = True
nodes = mat.node_tree.nodes
nodes.clear()
emission = nodes.new('ShaderNodeEmission')
emission.inputs['Color'].default_value = [r, g, b, 1.0]
output = nodes.new('ShaderNodeOutputMaterial')
mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
```

#### 5.3.7 抗锯齿设置 (中等)

**位置**: `render_images.py` 第513, 518, 558行

```python
# 原代码
old_use_antialiasing = render_args.use_antialiasing  # ❌
render_args.use_antialiasing = False                  # ❌

# 修复 - Cycles/EEVEE中抗锯齿通过其他设置控制
# 对于Cycles，使用samples控制
bpy.context.scene.cycles.use_denoising = False
# 对于EEVEE，使用
bpy.context.scene.eevee.taa_render_samples = 1
```

#### 5.3.8 平面创建参数 (轻微)

**位置**: `render_images.py` 第264行

```python
# 原代码
bpy.ops.mesh.primitive_plane_add(radius=5)  # ❌

# 修复
bpy.ops.mesh.primitive_plane_add(size=10)   # ✅ size = 2 * radius
```

### 5.4 Blender版本演变时间线

```
Blender 2.78c (2017) ← CLEVR原始开发版本
     ↓
Blender 2.79 (2018) - 最后一个使用旧API的版本
     ↓
Blender 2.80 (2019) - ⚠️ 重大API重构
  - 移除BLENDER_RENDER引擎
  - 图层系统 → Collections
  - 对象选择API变更
  - 用户偏好API变更
     ↓
Blender 2.90-3.x (2020-2023)
     ↓
Blender 4.x (2023-2024)
     ↓
Blender 5.0 (当前) - 继承2.80+的所有变更
```

### 5.5 兼容性总结

| 严重程度 | 问题数量 | 描述 |
|----------|----------|------|
| 🔴 严重 | 7 | 代码完全无法运行，必须修复 |
| 🟡 中等 | 2 | 功能受限，建议修复 |
| 🟢 轻微 | 1 | 参数名变更，易修复 |

---

## 6. API 使用指南

### 6.1 图像生成

```bash
# 基本用法
cd image_generation
blender --background --python render_images.py -- --num_images 10

# GPU加速（需要CUDA）
blender --background --python render_images.py -- --num_images 10 --use_gpu 1

# 高分辨率渲染
blender --background --python render_images.py -- \
    --num_images 10 \
    --width 640 \
    --height 480 \
    --render_num_samples 1024

# 保存Blender文件
blender --background --python render_images.py -- \
    --num_images 10 \
    --save_blendfiles 1
```

### 6.2 问题生成

```bash
cd question_generation

# 基本用法
python generate_questions.py

# 指定输入输出
python generate_questions.py \
    --input_scene_file ../output/CLEVR_scenes.json \
    --output_questions_file ../output/CLEVR_questions.json

# 控制问题数量
python generate_questions.py \
    --templates_per_image 10 \
    --instances_per_template 1
```

### 6.3 场景合并

```bash
cd image_generation
python collect_scenes.py \
    --input_dir ../output/scenes/ \
    --output_file ../output/CLEVR_all_scenes.json
```

---

## 7. 数据格式说明

### 7.1 场景JSON结构

```json
{
  "info": {
    "date": "01/31/2026",
    "version": "1.0",
    "split": "train",
    "license": "Creative Commons Attribution (CC-BY 4.0)"
  },
  "scenes": [
    {
      "split": "train",
      "image_index": 0,
      "image_filename": "CLEVR_train_000000.png",
      "objects": [
        {
          "shape": "cube",
          "size": "large",
          "material": "rubber",
          "color": "red",
          "3d_coords": [1.5, -2.0, 0.7],
          "rotation": 45.0,
          "pixel_coords": [160, 120, 0.85]
        }
      ],
      "relationships": {
        "left": [[1, 2], [2], []],
        "right": [[], [0], [0, 1]],
        "front": [[2], [], [0]],
        "behind": [[], [2], []]
      },
      "directions": {
        "behind": [-0.707, 0.707, 0.0],
        "front": [0.707, -0.707, 0.0],
        "left": [-0.707, -0.707, 0.0],
        "right": [0.707, 0.707, 0.0],
        "above": [0.0, 0.0, 1.0],
        "below": [0.0, 0.0, -1.0]
      }
    }
  ]
}
```

### 7.2 问题JSON结构

```json
{
  "info": {
    "date": "01/31/2026",
    "version": "1.0",
    "split": "train",
    "license": "Creative Commons Attribution (CC-BY 4.0)"
  },
  "questions": [
    {
      "image_index": 0,
      "image_filename": "CLEVR_train_000000.png",
      "question_index": 0,
      "question": "What size is the red cube?",
      "answer": "large",
      "question_family_index": 0,
      "program": [
        {"type": "scene", "inputs": []},
        {"type": "filter_color", "inputs": [0], "side_inputs": ["red"]},
        {"type": "filter_shape", "inputs": [1], "side_inputs": ["cube"]},
        {"type": "unique", "inputs": [2]},
        {"type": "query_size", "inputs": [3]}
      ]
    }
  ]
}
```

---

## 8. 迁移建议

### 8.1 方案一：使用Blender 2.79（推荐用于快速验证）

如果只需要生成数据而不需要最新Blender功能，建议使用 Blender 2.79：

```bash
# 下载 Blender 2.79b
wget https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2

# 解压并使用
tar -xjf blender-2.79b-linux-glibc219-x86_64.tar.bz2
./blender-2.79b-linux-glibc219-x86_64/blender --background --python render_images.py -- --num_images 10
```

### 8.2 方案二：完整迁移到Blender 5.0

如需使用 Blender 5.0，需要修改以下文件：

#### 需要修改的文件清单

1. **`utils.py`** - 约15处修改
   - 对象选择API
   - 活动对象设置
   - 图层系统重构

2. **`render_images.py`** - 约20处修改
   - GPU设置API
   - 渲染引擎设置
   - 抗锯齿设置
   - 材质系统
   - Cycles设置

3. **`data/base_scene.blend`** - 需要重新创建
   - 在Blender 5.0中重建场景
   - 更新灯光和相机设置

4. **`data/materials/*.blend`** - 需要重新创建
   - 重建所有材质节点
   - 使用现代Cycles/EEVEE shader

5. **`data/shapes/*.blend`** - 可能需要更新
   - 检查模型兼容性

### 8.3 迁移工作量估计

| 任务 | 难度 | 预估工作量 |
|------|------|-----------|
| Python API更新 | 中等 | 约100行代码修改 |
| 测试和调试 | 高 | 反复验证各功能 |
| .blend文件重建 | 中等 | 重建场景和材质 |
| 文档更新 | 低 | 更新使用说明 |

### 8.4 第三方替代方案

如果迁移成本过高，可以考虑：

1. **使用Docker容器**
   - 打包旧版Blender环境
   - 无需修改代码

2. **使用其他渲染工具**
   - Three.js (JavaScript)
   - PyVista / VTK (Python)
   - ModernGL (Python OpenGL)

---

## 附录

### A. 许可证信息

- **类型**: BSD License
- **发布方**: Facebook, Inc. (现Meta)
- **年份**: 2017-present

### B. 相关链接

- CLEVR 数据集官网: http://cs.stanford.edu/people/jcjohns/clevr/
- 原始GitHub仓库: https://github.com/facebookresearch/clevr-dataset-gen
- CLEVR-IEP (基线模型): https://github.com/facebookresearch/clevr-iep

### C. 联系方式

如有问题，请参考原项目的 GitHub Issues 或 CONTRIBUTING.md。

---

*报告生成时间: 2026年1月31日*
