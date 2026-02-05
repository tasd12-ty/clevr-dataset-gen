# ORDINAL-SPATIAL 数据集划分说明

## 📊 数据集规模分级（大小）

8卡GPU脚本提供4个规模预设，用于不同的使用场景：

| 规模 | train | val | test_iid | test_comp | test_hard | **总计** | **预计耗时** | **用途** |
|------|-------|-----|----------|-----------|-----------|---------|------------|----------|
| **tiny** | 8 | 8 | 8 | 8 | 8 | **40** | ~5分钟 | 快速测试流程 |
| **small** | 800 | 160 | 400 | 80 | 80 | **1,520** | ~2小时 | 验证和调试 |
| **medium** | 8,000 | 1,600 | 4,000 | 800 | 800 | **15,200** | ~20小时 | 中等规模研究 |
| **large** | 80,000 | 16,000 | 40,000 | 8,000 | 8,000 | **152,000** | ~200小时 | 完整基准数据集 |

### 使用示例

```bash
# 快速测试（推荐首次运行）
python scripts/build_8gpu.py --output ./data/test --size tiny

# 验证流程
python scripts/build_8gpu.py --output ./data/small --size small

# 完整数据集
python scripts/build_8gpu.py --output ./data/full --size large
```

---

## 🎯 数据集难度分级（Split类型）

每个规模都包含5个不同难度的分割（split），用于评估模型的不同能力：

### 1. train（训练集）
- **物体数量**: 4-10个
- **tau阈值**: 0.10（标准）
- **用途**: 模型训练

### 2. val（验证集）
- **物体数量**: 4-10个
- **tau阈值**: 0.10（标准）
- **分布**: 与训练集相同（IID）
- **用途**: 超参数调优、模型选择

### 3. test_iid（独立同分布测试集）
- **物体数量**: 4-10个
- **tau阈值**: 0.10（标准）
- **分布**: 与训练集相同（IID）
- **用途**: 标准泛化能力评估

### 4. test_comp（组合泛化测试集）
- **物体数量**: **10-15个**（更多！）
- **tau阈值**: 0.10（标准）
- **难度提升**: 更多物体导致更复杂的空间关系
- **用途**: 评估模型在更复杂场景下的泛化能力

### 5. test_hard（困难测试集）
- **物体数量**: 4-10个
- **tau阈值**: **0.05**（更严格！）
- **难度提升**: 更小的容差要求更精确的判断
- **用途**: 评估模型的精细空间判断能力

---

## ⚙️ 容差参数（tau）详解

容差参数 `tau` 控制比较的敏感度，影响约束的严格程度。

### 数学定义

对于两个值 `a` 和 `b`，带容差的比较定义为：

- **小于**: `a <_tau b` ⟺ `a < b * (1 - tau)`
- **约等**: `a ~=_tau b` ⟺ `|a - b| <= tau * max(a, b)`
- **大于**: `a >_tau b` ⟺ `a > b * (1 + tau)`

### 预设值

| 预设 | tau值 | 描述 | 使用场景 |
|------|-------|------|----------|
| **严格** (strict) | 0.05 | 高精度要求 | test_hard 分割 |
| **标准** (standard) | 0.10 | 平衡准确性与鲁棒性 | train/val/test_iid/test_comp |
| **宽松** (relaxed) | 0.20 | 更容忍误差 | 噪声场景（未使用）|

### 实例

假设有两个距离：`d1 = 10.0`, `d2 = 11.0`

**tau = 0.10 (标准)**
- `d1 <_0.10 d2` ? → `10.0 < 11.0 * 0.9 = 9.9` ? → **否**
- `d1 ~=_0.10 d2` ? → `|10.0 - 11.0| <= 0.10 * 11.0 = 1.1` ? → **是** ✓
- 结论: `d1 ~= d2` (约等)

**tau = 0.05 (严格)**
- `d1 <_0.05 d2` ? → `10.0 < 11.0 * 0.95 = 10.45` ? → **是** ✓
- 结论: `d1 < d2` (小于)

→ **更小的tau使比较更敏感**，相同的数据会产生更多"<"或">"而非"~="判断。

---

## 📈 约束难度等级

每个约束的难度由度量比率决定：

| 等级 | 比率范围 | 描述 | 示例 |
|------|----------|------|------|
| **1** | > 2.0 | 简单 | `dist(A,B)=2, dist(C,D)=5` → 比率=2.5 |
| **2** | 1.5 - 2.0 | 中等偏易 | `dist(A,B)=3, dist(C,D)=5` → 比率=1.67 |
| **3** | 1.3 - 1.5 | 中等 | `dist(A,B)=3.5, dist(C,D)=5` → 比率=1.43 |
| **4** | 1.15 - 1.3 | 中等偏难 | `dist(A,B)=4, dist(C,D)=5` → 比率=1.25 |
| **5** | 1.05 - 1.15 | 困难 | `dist(A,B)=4.5, dist(C,D)=5` → 比率=1.11 |
| **6** | 1.0 - 1.05 | 极难 | `dist(A,B)=4.9, dist(C,D)=5` → 比率=1.02 |

### 计算公式

```python
# 对于 QRR 约束: dist(A,B) vs dist(C,D)
ratio = max(dist_AB, dist_CD) / min(dist_AB, dist_CD)

if ratio > 2.0:
    difficulty_level = 1
elif ratio > 1.5:
    difficulty_level = 2
# ... 以此类推
```

---

## 🔍 约束类型概览

数据集包含7种类型的空间约束：

| 约束类型 | 符号 | 描述 | 示例 |
|----------|------|------|------|
| **QRR** | `dist(A,B) < dist(C,D)` | 四元相对距离关系 | "红色立方体和蓝色球之间的距离小于绿色圆锥和黄色柱之间的距离" |
| **TRR** | `hour(A; B→C) = 3` | 三元时钟方向关系 | "红色立方体在蓝色球→绿色圆锥轴的3点钟方向" |
| **Axial** | `left_of(A, B)` | 二元轴向偏序 | "A在B的左边" |
| **Topology** | `disjoint(A, B)` | RCC-8拓扑关系 | "A和B不接触" |
| **Occlusion** | `occludes(A, B)` | 遮挡关系 | "A遮挡了B" |
| **Size** | `bigger(A, B)` | 尺寸比较 | "A比B大" |
| **Closer** | `closer(A; B, C)` | 三元距离比较 | "B比C更接近A" |

---

## 📁 输出数据结构

每个场景包含完整的对应关系：

```
data/benchmark_small/
├── images/
│   ├── single_view/
│   │   └── train_00000.png          # 单视角图片
│   └── multi_view/
│       └── train_00000/
│           ├── view_0.png           # 0° (正面)
│           ├── view_1.png           # 90° (右侧)
│           ├── view_2.png           # 180° (背面)
│           └── view_3.png           # 270° (左侧)
│
├── metadata/
│   └── train_00000.json             # 场景元数据+约束
│       {
│         "scene_id": "train_00000",
│         "objects": [...],          # 物体位置、颜色、大小
│         "constraints": {           # 真值约束
│           "qrr": [...],
│           "trr": [...],
│           "axial": [...],
│           ...
│         },
│         "tau": 0.10,
│         "n_objects": 7
│       }
│
├── splits/
│   ├── train.json                   # 训练集索引
│   ├── val.json                     # 验证集索引
│   ├── test_iid.json               # IID测试集索引
│   ├── test_comp.json              # 组合测试集索引
│   └── test_hard.json              # 困难测试集索引
│
└── dataset_info.json                # 数据集元信息
    {
      "name": "ORDINAL-SPATIAL Benchmark",
      "created": "2026-02-05T10:30:00",
      "n_gpus": 8,
      "render_quality": {...},
      "splits": {
        "train": 800,
        "val": 160,
        ...
      },
      "statistics": {...}
    }
```

---

## 🎨 渲染质量分级

| 预设 | 分辨率 | 采样数 | 预计耗时/场景 | 用途 |
|------|--------|--------|--------------|------|
| **draft** | 480×320 | 64 | ~2秒 | 快速测试 |
| **normal** | 1024×768 | 256 | ~5秒 | 标准质量（默认）|
| **high** | 1024×768 | 512 | ~10秒 | 高质量渲染 |

### 使用示例

```bash
# 草稿质量（快速）
python scripts/build_8gpu.py --output ./data/test --size tiny --quality draft

# 高质量渲染
python scripts/build_8gpu.py --output ./data/full --size large --quality high
```

---

## 📊 完整配置矩阵

### 规模 × 质量

| 规模/质量 | draft (64 samples) | normal (256 samples) | high (512 samples) |
|-----------|-------------------|---------------------|-------------------|
| **tiny (40)** | ~2分钟 | ~5分钟 | ~10分钟 |
| **small (1.5K)** | ~1小时 | ~2小时 | ~4小时 |
| **medium (15K)** | ~8小时 | ~20小时 | ~40小时 |
| **large (152K)** | ~80小时 | ~200小时 | ~400小时 |

### Split × 难度

| Split | 物体数 | tau | 难度因素 |
|-------|--------|-----|----------|
| train | 4-10 | 0.10 | 标准 |
| val | 4-10 | 0.10 | 标准 |
| test_iid | 4-10 | 0.10 | 标准 |
| test_comp | **10-15** | 0.10 | **更多物体** |
| test_hard | 4-10 | **0.05** | **更严格tau** |

---

## 🚀 快速参考

### 推荐工作流

```bash
# 1. 快速测试（5分钟）
python scripts/build_8gpu.py --output ./data/test --size tiny

# 2. 验证流程（2小时，normal质量）
python scripts/build_8gpu.py --output ./data/small --size small

# 3. 完整数据集（200小时，high质量）
python scripts/build_8gpu.py --output ./data/full --size large --quality high
```

### 数据集验证

```bash
# 检查数据完整性
python -m ordinal_spatial.scripts.validate_benchmark \
    --dataset-dir ./data/benchmark_small

# 查看数据集信息
cat ./data/benchmark_small/dataset_info.json | jq
```

---

## ❓ 常见问题

### Q1: tiny和small规模有什么区别？
- **tiny**: 40个场景，用于快速测试脚本是否能正常运行（约5分钟）
- **small**: 1,520个场景，用于验证完整流程和初步实验（约2小时）

### Q2: test_comp比test_iid难在哪里？
- **test_comp**包含10-15个物体（vs test_iid的4-10个）
- 更多物体意味着：
  - 更多的成对关系（n²增长）
  - 更复杂的遮挡和空间布局
  - 更难的约束一致性检查

### Q3: tau=0.05和tau=0.10的实际区别？
- **tau=0.10**: 对于10.0 vs 11.0，判断为"约等" (~=)
- **tau=0.05**: 对于10.0 vs 11.0，判断为"小于" (<)
- 更小的tau要求模型做出更精细的判断

### Q4: 为什么large规模需要200小时？
- 152,000个场景 × 5张图片/场景 = 760,000张图片
- 即使用8卡GPU并行，每张图片约需10秒（normal质量）
- 建议使用`--quality draft`可减半至~80小时

---

## 📚 相关文档

- [CLAUDE.md](../CLAUDE.md) - 仓库完整指南
- [scripts/README.md](./README.md) - 8卡GPU脚本使用说明
- [ordinal_spatial/README.md](../ordinal_spatial/README.md) - ORDINAL-SPATIAL框架文档
