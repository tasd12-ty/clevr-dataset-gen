# 多GPU数据集生成脚本

## 快速开始

```bash
# 测试运行（约5分钟，40个场景）
python scripts/build_8gpu.py --output ./data/test --size tiny

# 小规模（约2小时，1520个场景）
python scripts/build_8gpu.py --output ./data/small --size small

# 完整数据集（约200小时，152,000个场景）
python scripts/build_8gpu.py --output ./data/full --size large --quality high
```

## 参数说明

### 必需参数
- `--output`: 输出目录

### 规模预设（--size）
| 预设 | 总场景数 | 预计耗时 | 用途 |
|------|---------|---------|------|
| tiny | 40 | 5分钟 | 快速测试 |
| small | 1,520 | 2小时 | 验证流程 |
| medium | 15,200 | 20小时 | 中等规模 |
| large | 152,000 | 200小时 | 完整数据集 |

### 质量预设（--quality）
| 预设 | 分辨率 | 采样数 | 说明 |
|------|--------|--------|------|
| draft | 480×320 | 64 | 草稿质量 |
| normal | 1024×768 | 256 | 标准质量（默认）|
| high | 1024×768 | 512 | 高质量 |

### 高级参数
- `--n-gpus`: GPU数量（默认8）
- `--blender`: Blender路径（默认WSL路径）

## 输出结构

```
output/
├── images/
│   ├── single_view/     # 单视角图片
│   │   └── scene_*.png
│   └── multi_view/      # 多视角图片
│       └── scene_*/
│           ├── view_0.png
│           ├── view_1.png
│           ├── view_2.png
│           └── view_3.png
├── metadata/            # 场景元数据+约束
│   └── scene_*.json
├── splits/              # 数据集分割
│   ├── train.json
│   ├── val.json
│   ├── test_iid.json
│   ├── test_comp.json
│   └── test_hard.json
└── dataset_info.json    # 数据集信息
```

## 数据对应关系

每个场景保证以下对应：
- **scene_id**: 唯一标识符（如 `train_00000`）
- **图片**: `images/single_view/{scene_id}.png` 和 `images/multi_view/{scene_id}/view_*.png`
- **元数据**: `metadata/{scene_id}.json` 包含：
  - 物体位置、属性
  - 相机参数
  - 真值约束（QRR, TRR, 拓扑等）
- **分割文件**: `splits/*.json` 中引用正确的路径

## 工作原理

1. **任务分配**: 将场景均匀分配到8个GPU
2. **并行渲染**: 每个GPU独立渲染（通过`CUDA_VISIBLE_DEVICES`隔离）
3. **约束提取**: 从Blender场景数据提取真值约束
4. **结果合并**: 整合所有GPU输出到统一数据集
5. **验证**: 确保所有数据文件正确对应

## 代码结构

```
scripts/
├── build_8gpu.py              # 用户接口（简洁）
├── lib/
│   ├── multi_gpu_builder.py   # 核心构建逻辑
│   ├── gpu_worker.py          # GPU工作进程
│   └── merger.py              # 结果合并
└── README.md                  # 本文档
```

## 故障排查

### GPU内存不足
- 降低质量: `--quality draft`
- 减少GPU数量: `--n-gpus 4`

### 渲染失败
- 检查Blender路径是否正确
- 查看日志中的错误信息
- 尝试tiny规模测试

### 数据不对应
- 脚本自动确保对应关系
- 检查 `dataset_info.json` 中的统计信息
- 验证元数据文件是否存在
