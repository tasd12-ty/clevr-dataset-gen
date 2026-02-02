# CLAUDE.md - AI 助手指南 / AI Assistant Guide

## 仓库概述 / Repository Overview

本仓库包含用于生成 **CLEVR（组合语言和基础视觉推理）数据集** 和 **ORDINAL-SPATIAL 空间推理基准** 的代码。主要功能：

1. 使用 Blender 渲染合成 3D 场景（支持 2.78 - 5.0+）
2. 生成组合式自然语言问题及函数程序
3. 评估视觉-语言模型（VLM）的序空间推理能力
4. **VLM 约束提取智能体**（Task-1/2/3）

---

This repository contains code for generating the **CLEVR (Compositional Language and Elementary Visual Reasoning) dataset** and the **ORDINAL-SPATIAL spatial reasoning benchmark**. It provides tools for:

1. Rendering synthetic 3D scenes using Blender (supports 2.78 - 5.0+)
2. Generating compositional natural language questions with functional programs
3. Evaluating Vision-Language Models (VLMs) on ordinal spatial reasoning tasks
4. **VLM Constraint Extraction Agents** (Task-1/2/3)

## Directory Structure

```
clevr-dataset-gen/
├── image_generation/          # Blender-based image rendering
│   ├── render_images.py       # Main rendering script (Blender 2.78-5.0 compatible)
│   ├── utils.py               # Blender utilities (version-aware)
│   ├── collect_scenes.py      # Scene collection utility
│   ├── create_base_scene_blender5.py    # Generate v5 base scene
│   ├── create_materials_blender5.py     # Generate v5 materials
│   ├── create_shapes_blender5.py        # Generate v5 shapes
│   ├── setup_blender5.sh      # One-click Blender 5.0 setup
│   └── data/
│       ├── base_scene.blend   # Base scene (Blender 2.78)
│       ├── base_scene_v5.blend # Base scene (Blender 5.0+)
│       ├── materials/         # Materials (Blender 2.78)
│       ├── materials_v5/      # Materials (Blender 5.0+)
│       ├── shapes/            # Shapes (Blender 2.78)
│       ├── shapes_v5/         # Shapes (Blender 5.0+)
│       └── properties.json    # Object property definitions
│
├── question_generation/       # Question synthesis
│   ├── generate_questions.py  # Main question generation script
│   ├── question_engine.py     # Template processing engine
│   ├── metadata.json          # Functional programming language specs
│   ├── synonyms.json          # Natural language synonyms
│   └── CLEVR_1.0_templates/   # Question templates (~40 files)
│
├── ordinal_spatial/           # 空间推理基准模块 / Spatial reasoning benchmark
│   ├── dsl/                   # 领域特定语言 / Domain-Specific Language
│   ├── evaluation/            # 评估指标 / Metrics and consistency checking
│   ├── generation/            # 约束生成 / Constraint and dataset generation
│   ├── baselines/             # 基线模型 / Baseline implementations
│   ├── prompts/               # VLM 提示模板 / VLM prompt templates
│   ├── tasks/                 # 任务运行器 / Evaluation task runners
│   ├── agents/                # VLM 约束提取智能体 / Constraint extraction agents
│   │   ├── base.py            # 基类定义 / Base classes
│   │   ├── vlm_constraint_agent.py    # VLM 智能体 (Task-2/3)
│   │   ├── blender_constraint_agent.py # Blender 智能体 (Task-1)
│   │   ├── cli.py             # 命令行接口 / CLI
│   │   ├── prompts/           # 智能体提示词 / Agent prompts
│   │   └── tests/             # 智能体测试 / Agent tests
│   ├── scripts/               # 命令行工具 / CLI tools
│   └── tests/                 # 单元测试 / Unit tests
│
└── images/                    # Example rendered images
```

## Blender Compatibility

| Version | Status | Resource Files | Notes |
|---------|--------|----------------|-------|
| 2.78c | Fully Supported | `data/*.blend`, `data/materials/`, `data/shapes/` | Original version |
| 2.79b | Fully Supported | Same as 2.78 | Last legacy API version |
| 2.80-3.x | Supported | `data/*_v5.blend`, `data/materials_v5/`, `data/shapes_v5/` | New API |
| 4.x | Supported | Same as v5 | |
| **5.0+** | **Supported** | Same as v5 | Tested with 5.0.1 |

The code uses `IS_BLENDER_280_OR_LATER` flag to automatically select the correct API.

## Development Workflows

### Image Generation (Blender 5.0+)

```bash
cd image_generation

# Step 1: Generate compatible resource files (one-time setup)
./setup_blender5.sh /path/to/blender
# Or manually:
blender --background --python create_base_scene_blender5.py
blender --background --python create_materials_blender5.py
blender --background --python create_shapes_blender5.py

# Step 2: Add Python path (one-time setup)
# Linux:
echo $PWD >> ~/.config/blender/5.0/python/lib/python3.11/site-packages/clevr.pth

# Step 3: Render images
blender --background --python render_images.py -- \
    --base_scene_blendfile data/base_scene_v5.blend \
    --material_dir data/materials_v5 \
    --shape_dir data/shapes_v5 \
    --num_images 10

# With GPU acceleration
blender --background --python render_images.py -- \
    --base_scene_blendfile data/base_scene_v5.blend \
    --material_dir data/materials_v5 \
    --shape_dir data/shapes_v5 \
    --num_images 10 \
    --use_gpu 1
```

### Image Generation (Blender 2.78-2.79 Legacy)

```bash
cd image_generation

# Setup
echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth

# Render
blender --background --python render_images.py -- --num_images 10
```

### Question Generation

```bash
cd question_generation
python generate_questions.py \
    --input_scene_file ../output/CLEVR_scenes.json \
    --output_questions_file ../output/CLEVR_questions.json
```

### ORDINAL-SPATIAL Module

```bash
# Install dependencies
pip install -r ordinal_spatial/requirements.txt

# Generate dataset
python -m ordinal_spatial.scripts.generate_dataset --small --output-dir ./data

# Run baseline evaluation
python -m ordinal_spatial.scripts.run_baseline \
    --baseline oracle \
    --task t1-q \
    --data ./data \
    --split test_iid
```

### VLM 约束提取智能体 / VLM Constraint Agent (Task-1/2/3)

```bash
# Task-3: 单视角约束提取 / Single-view extraction
python -m ordinal_spatial.agents.cli extract \
    --image scene.png \
    --output constraints.json \
    --tau 0.10

# Task-2: 多视角约束提取 / Multi-view extraction
python -m ordinal_spatial.agents.cli extract \
    --images view1.png view2.png view3.png \
    --output constraints.json

# 使用自定义模型 / With custom model
python -m ordinal_spatial.agents.cli extract \
    --image scene.png \
    --model openai/gpt-4o \
    --output constraints.json
```

#### Task-1: Blender 真值提取 / Ground Truth Extraction

```python
from ordinal_spatial.agents import BlenderConstraintAgent

agent = BlenderConstraintAgent()

# 从 CLEVR 场景文件提取 / Extract from CLEVR scenes
constraints = agent.extract_from_clevr_scenes(
    "output/CLEVR_scenes.json",
    tau=0.10
)

# 从 .blend 文件提取 / Extract from .blend file
constraints = agent.extract_from_blend_file("scene.blend", tau=0.10)
```

#### Task-2/3: VLM 约束提取 / VLM Extraction

```python
from ordinal_spatial.agents import VLMConstraintAgent, VLMAgentConfig

config = VLMAgentConfig(model="google/gemma-3-27b-it")
agent = VLMConstraintAgent(config)

# 单视角 / Single view
result = agent.extract_from_single_view("scene.png", tau=0.10)

# 多视角 / Multi view
result = agent.extract_from_multi_view(
    ["view1.png", "view2.png", "view3.png"],
    tau=0.10
)
```

## Key API Changes for Blender 5.0

| Blender 2.79 | Blender 5.0 | Location |
|--------------|-------------|----------|
| `obj.select = True` | `obj.select_set(True)` | utils.py |
| `scene.objects.active = obj` | `view_layer.objects.active = obj` | utils.py |
| `user_preferences` | `preferences` | render_images.py |
| `BLENDER_RENDER` engine | Use `CYCLES` with emission shaders | render_images.py |
| `obj.layers[i]` | Collections system | utils.py |
| `mat.use_shadeless = True` | Emission shader nodes | render_images.py |
| `primitive_plane_add(radius=)` | `primitive_plane_add(size=)` | render_images.py |
| `matrix * vector` | `matrix @ vector` | render_images.py |
| `bpy.ops.material.new()` | `bpy.data.materials.new()` | utils.py |
| Node name `'Material Output'` | Use type `'OUTPUT_MATERIAL'` | utils.py |

## Code Conventions

### Style Guidelines
- **Indentation**: 2 spaces (not tabs)
- **Line length**: 80 characters max
- **Naming**: snake_case for functions, PascalCase for classes

### Version-Aware Code Pattern

```python
# At module level
BLENDER_VERSION = bpy.app.version
IS_BLENDER_280_OR_LATER = BLENDER_VERSION >= (2, 80, 0)

# In functions
if IS_BLENDER_280_OR_LATER:
    # Blender 2.80+ / 5.0 code
    obj.select_set(True)
else:
    # Blender 2.79 and earlier
    obj.select = True
```

### Object Name Compatibility

```python
def get_object_by_name(name, alternative_names=None):
    """Get object with fallback to alternative names."""
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    if alternative_names:
        for alt_name in alternative_names:
            if alt_name in bpy.data.objects:
                return bpy.data.objects[alt_name]
    raise KeyError(f"Object not found: {name}")

# Usage
lamp_key = get_object_by_name('Lamp_Key', ['Light_Key', 'Key'])
```

## Environment Configuration

### Blender Path (WSL)

在当前 WSL 环境中，Blender 可执行文件位于：

```
/mnt/d/tools/blender/blender.exe
```

使用示例：

```bash
# 检查 Blender 版本
/mnt/d/tools/blender/blender.exe --version

# 渲染图像
/mnt/d/tools/blender/blender.exe --background --python render_images.py -- --num_images 10

# 生成 Benchmark 数据集
python -m ordinal_spatial.scripts.build_benchmark \
    --output-dir ./data/benchmark_test \
    --blender-path /mnt/d/tools/blender/blender.exe \
    --tiny
```

## Dependencies

### External Tools
- **Blender 2.78-5.0+**: Required for image rendering (WSL: `/mnt/d/tools/blender/blender.exe`)
- **Python 3.5+**: For original CLEVR code
- **Python 3.7+**: For ordinal_spatial module

### Python Packages (ordinal_spatial)
```
numpy>=1.21.0
scipy>=1.7.0
Pillow>=9.0.0
pydantic>=2.0.0
networkx>=2.6.0
torch>=2.0.0
openai>=1.0.0
pytest>=7.0.0
```

## Common Tasks for AI Assistants

### Adding Support for New Blender Version

1. Test module import in new Blender version
2. Check for API deprecation warnings
3. Update `IS_BLENDER_280_OR_LATER` checks if needed
4. Update resource generation scripts if needed
5. Test rendering pipeline end-to-end

### Modifying Scene Generation

1. Edit `image_generation/render_images.py` for rendering changes
2. For Blender 5.0+, may need to update `create_*_blender5.py` scripts
3. Test with both legacy and modern Blender versions

### Debugging Blender Issues

```bash
# Check Blender version
blender --version

# Test module import
blender --background --python-expr "import bpy; print(bpy.app.version)"

# Test resource loading
blender --background --python-expr "
import bpy, sys
sys.path.insert(0, '/path/to/image_generation')
import utils
print('Success!')
"
```

## 关键文件参考 / Key Files Reference

| File | Purpose |
|------|---------|
| `image_generation/render_images.py` | Main rendering script (version-aware) |
| `image_generation/render_multiview.py` | Multi-view rendering with spherical cameras |
| `image_generation/utils.py` | Blender utilities (version-aware) |
| `image_generation/create_base_scene_blender5.py` | Generate Blender 5.0 base scene |
| `image_generation/create_materials_blender5.py` | Generate Blender 5.0 materials |
| `image_generation/create_shapes_blender5.py` | Generate Blender 5.0 shapes |
| `question_generation/generate_questions.py` | Question synthesis engine |
| `ordinal_spatial/dsl/schema.py` | Core data models |
| `ordinal_spatial/scripts/build_benchmark.py` | Benchmark dataset builder |
| `ordinal_spatial/scripts/validate_benchmark.py` | Benchmark validation tool |
| `ordinal_spatial/agents/vlm_constraint_agent.py` | VLM constraint extraction (Task-2/3) |
| `ordinal_spatial/agents/cli.py` | Agent command-line interface |

## ORDINAL-SPATIAL Benchmark Generation

### Quick Start

```bash
# Generate tiny dataset for testing
python -m ordinal_spatial.scripts.build_benchmark \
    --output-dir ./data/benchmark_tiny \
    --blender-path /mnt/d/tools/blender/blender.exe \
    --tiny

# Generate small dataset
python -m ordinal_spatial.scripts.build_benchmark \
    --output-dir ./data/benchmark_small \
    --blender-path /mnt/d/tools/blender/blender.exe \
    --small

# Generate full benchmark
python -m ordinal_spatial.scripts.build_benchmark \
    --output-dir ./data/benchmark_full \
    --blender-path /mnt/d/tools/blender/blender.exe \
    --n-train 1000 --n-val 200 --n-test 500 \
    --n-test-comp 200 --n-test-hard 200

# Validate generated dataset
python -m ordinal_spatial.scripts.validate_benchmark \
    --dataset-dir ./data/benchmark_tiny
```

### Dataset Splits

| Split | Objects | Tau | Description |
|-------|---------|-----|-------------|
| train | 4-10 | 0.10 | Training set |
| val | 4-10 | 0.10 | Validation set |
| test_iid | 4-10 | 0.10 | IID test set |
| test_comp | 10-15 | 0.10 | Compositional (more objects) |
| test_hard | 4-10 | 0.05 | Stricter tau threshold |

### Multi-View Rendering

Each scene generates:
- 1 single-view image (view_0)
- 4 multi-view images (view_0, view_1, view_2, view_3) at 90° azimuth intervals

Camera configuration:
- Distance: 12.0 units
- Elevation: 30°
- Azimuths: 45°, 135°, 225°, 315°

### Constraint Types Extracted

| Type | Description |
|------|-------------|
| QRR | Quantitative Ratio Relations (distance comparisons) |
| TRR | Ternary Reference Relations (clock-face directions) |
| Topology | Disjoint/touching relationships |
| Occlusion | Which objects occlude others |
| Axial | Left/right/front/behind relations |
| Size | Bigger/smaller comparisons |
| Closer | Distance ordering from anchor objects |

### Dense Scene Placement (10-15 objects)

For scenes with many objects, placement parameters scale automatically:

| Objects | Placement Area | min_dist | margin |
|---------|----------------|----------|--------|
| 1-6 | 6×6 | 0.25 | 0.40 |
| 7-10 | 7×7 | 0.25 | 0.40 |
| 11-15 | 8×8 | 0.175 | 0.28 |

## Blender 5.0 Key Fixes (vs Original CLEVR)

### 1. Object Grounding Fix

**Problem**: Objects were floating above the ground plane.

**Root Cause**: Original CLEVR shapes had origin at center; v5 shapes have origin at bottom.

**Fix Location**: `image_generation/utils.py:154-156`

```python
# Original CLEVR (origin at center):
bpy.ops.transform.translate(value=(x, y, scale))

# v5 Fix (origin at bottom):
bpy.ops.transform.translate(value=(x, y, 0))
```

**Related Files**:
- `create_shapes_blender5.py`: Sets origin to bottom via `origin_set(type='ORIGIN_CURSOR')`
- `create_base_scene_blender5.py`: Ground plane at z=0

### 2. Recursion Limit Fix

**Problem**: Infinite recursion when placing objects due to occlusion retry loop.

**Fix Location**: `image_generation/render_images.py:399-407`

```python
def add_random_objects(scene_struct, num_objects, args, camera, _retry_count=0):
  MAX_OCCLUSION_RETRIES = 50
  if _retry_count >= MAX_OCCLUSION_RETRIES:
    raise RuntimeError("Failed to place objects after 50 attempts")
```

### 3. Graceful Scene Failure Handling

**Problem**: Single scene failure crashed entire batch.

**Fix Location**: `image_generation/render_images.py:195-225`

```python
try:
  render_scene(args, ...)
  successful_scenes += 1
except RuntimeError as e:
  print(f"Warning: Failed to render scene {i}: {e}")
  failed_scenes += 1
  continue
```

### 4. Skip Visibility Check Option

**Problem**: Visibility check is slow and often fails.

**Fix**: Set `--min_pixels_per_object 0` to skip visibility checking.

**Fix Location**: `image_generation/render_images.py:565-567`

```python
if min_pixels_per_object <= 0:
  return True  # Skip visibility check
```

### 5. Material Node Localization Fix

**Problem**: Node name `'Material Output'` is localized (e.g., '材质输出' in Chinese).

**Fix Location**: `image_generation/utils.py:197-199`

```python
# Find by type instead of name
for n in mat.node_tree.nodes:
  if n.type == 'OUTPUT_MATERIAL':
    output_node = n
```

## Troubleshooting

### ImportError: cannot import utils
Ensure `clevr.pth` is in the correct Blender Python site-packages directory.

### GPU Rendering Issues
1. Verify GPU drivers
2. Check `--use_gpu 1` flag
3. Code tries CUDA → OptiX → HIP → OneAPI automatically

### Material/Node Errors in Blender 5.0
- Ensure using `data/materials_v5/` directory
- Node names may be localized; code finds nodes by type, not name
- Materials need `use_fake_user = True` to be saved

### Blender Crashes
- Reduce `--render_num_samples` for testing
- Check memory availability
- Try CPU rendering first (`--use_gpu 0`)
