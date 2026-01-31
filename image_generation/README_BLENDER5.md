# Blender 5.0 兼容性指南

本文档说明如何在 Blender 5.0 / 4.x / 2.80+ 中运行 CLEVR 数据集生成工具。

## 快速开始

### 方法一：一键设置（推荐）

```bash
cd image_generation
./setup_blender5.sh /path/to/blender
```

这将自动创建所有必需的资源文件。

### 方法二：手动设置

1. 创建基础场景：
```bash
blender --background --python create_base_scene_blender5.py
```

2. 创建材质：
```bash
blender --background --python create_materials_blender5.py
```

3. 创建形状：
```bash
blender --background --python create_shapes_blender5.py
```

## 生成图像

使用新创建的资源文件渲染图像：

```bash
blender --background --python render_images.py -- \
    --base_scene_blendfile data/base_scene_v5.blend \
    --material_dir data/materials_v5 \
    --shape_dir data/shapes_v5 \
    --num_images 10
```

### GPU 加速

```bash
blender --background --python render_images.py -- \
    --base_scene_blendfile data/base_scene_v5.blend \
    --material_dir data/materials_v5 \
    --shape_dir data/shapes_v5 \
    --num_images 10 \
    --use_gpu 1
```

支持的 GPU 类型：
- NVIDIA CUDA
- NVIDIA OptiX
- AMD HIP
- Intel OneAPI

## Python 路径配置

如果遇到 `ImportError: cannot import utils` 错误，需要配置 Python 路径：

```bash
# 找到 Blender Python 的 site-packages 目录
# Linux 示例
echo $PWD >> ~/.config/blender/5.0/python/lib/python3.11/site-packages/clevr.pth

# macOS 示例
echo $PWD >> /Applications/Blender.app/Contents/Resources/5.0/python/lib/python3.11/site-packages/clevr.pth
```

## 主要 API 变更

| Blender 2.79 | Blender 5.0 | 说明 |
|--------------|-------------|------|
| `obj.select = True` | `obj.select_set(True)` | 对象选择 |
| `scene.objects.active` | `view_layer.objects.active` | 活动对象 |
| `user_preferences` | `preferences` | 用户偏好 |
| `BLENDER_RENDER` | `BLENDER_EEVEE` / `CYCLES` | 渲染引擎 |
| `obj.layers[i]` | Collections | 图层系统 |
| `mat.use_shadeless` | Emission shader | 无阴影材质 |
| `primitive_plane_add(radius=)` | `primitive_plane_add(size=)` | 创建平面 |
| `matrix * vector` | `matrix @ vector` | 矩阵运算 |

## 文件结构

```
image_generation/
├── render_images.py          # 主渲染脚本 (已更新)
├── utils.py                  # 工具函数 (已更新)
├── create_base_scene_blender5.py    # 创建基础场景
├── create_materials_blender5.py     # 创建材质
├── create_shapes_blender5.py        # 创建形状
├── setup_blender5.sh         # 一键设置脚本
├── data/
│   ├── base_scene.blend      # 原始场景 (Blender 2.78)
│   ├── base_scene_v5.blend   # 新场景 (Blender 5.0)
│   ├── materials/            # 原始材质
│   ├── materials_v5/         # 新材质
│   ├── shapes/               # 原始形状
│   └── shapes_v5/            # 新形状
```

## 故障排除

### 1. "Object not found: Lamp_Key"

基础场景中的灯光命名不匹配。代码已更新支持多种命名约定：
- `Lamp_Key` / `Light_Key` / `Key` / `KeyLight`

使用 `setup_blender5.sh` 创建的场景使用 `Lamp_Key` 命名。

### 2. GPU 渲染失败

检查 GPU 驱动是否正确安装：
```bash
# NVIDIA
nvidia-smi

# 在 Blender 中检查
blender --background --python -c "import bpy; print(bpy.context.preferences.addons['cycles'].preferences.get_devices())"
```

### 3. 渲染结果与原版不同

Blender 5.0 的 Cycles 渲染器有显著改进，渲染结果可能略有不同。如需完全一致的结果，请使用 Blender 2.79b。

## 版本兼容性

| Blender 版本 | 支持状态 |
|-------------|---------|
| 2.78c | 原始版本，完全支持 |
| 2.79b | 完全支持 |
| 2.80 - 2.93 | 需要新资源文件 |
| 3.x | 需要新资源文件 |
| 4.x | 需要新资源文件 |
| 5.0 | 需要新资源文件 |
