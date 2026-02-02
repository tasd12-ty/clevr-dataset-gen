# ORDINAL-SPATIAL 基准测试

一个通过比较性序关系约束来评估视觉-语言模型（VLM）空间理解能力的综合评估框架。

## 概述

ORDINAL-SPATIAL 通过以下方式评估 VLM 的空间推理能力：

- **QRR（四元相对关系）**：成对距离比较（例如："dist(A,B) < dist(C,D)?"）
- **TRR（三元时钟关系）**：基于时钟表盘的方向关系

### 任务类型

| 任务 | 描述 | 评估指标 |
|------|------|----------|
| **T1-Q** | QRR 分类任务 | 准确率、宏平均F1 |
| **T1-C** | TRR 分类任务 | 小时准确率、象限准确率 |
| **T2** | 约束提取任务 | 精确率、召回率、F1、一致性 |
| **T3** | 序重构任务 | NRMS、约束满足率 |

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 快速开始

### 生成数据集

```bash
# 生成小型测试数据集
python -m ordinal_spatial.scripts.generate_dataset --small --output-dir ./data

# 生成完整数据集
python -m ordinal_spatial.scripts.generate_dataset \
    --n-train 10000 \
    --n-val 1500 \
    --n-test 1500 \
    --output-dir ./data
```

### 运行基线模型

```bash
# 在 T1-Q 任务上运行 oracle 基线
python -m ordinal_spatial.scripts.run_baseline \
    --baseline oracle \
    --task t1-q \
    --data ./data \
    --split test_iid \
    --output ./results

# 运行 VLM 基线（需要 API 密钥）
export OPENROUTER_API_KEY="your-key"
python -m ordinal_spatial.scripts.run_baseline \
    --baseline vlm_direct \
    --task t2 \
    --data ./data \
    --model openai/gpt-4o
```

### 评估预测结果

```bash
python -m ordinal_spatial.scripts.evaluate \
    --predictions ./results/predictions.json \
    --ground-truth ./results/ground_truth.json \
    --task t1-q \
    --analyze-failures
```

### 可视化结果

```bash
python -m ordinal_spatial.scripts.visualize \
    --predictions ./results/predictions.json \
    --ground-truth ./results/ground_truth.json \
    --task t1-q \
    --output-dir ./visualizations
```

## API 使用

### DSL 核心

```python
from ordinal_spatial.dsl.comparators import compare, Comparator

# 使用容差比较距离
result = compare(a=10.0, b=11.5, tau=0.10)
print(result)  # Comparator.APPROX (~=)

# 严格小于
result = compare(a=8.0, b=12.0, tau=0.10)
print(result)  # Comparator.LT (<)
```

### QRR 约束

```python
from ordinal_spatial.dsl.predicates import compute_qrr, MetricType

objects = {
    "A": {"position": [0, 0, 0]},
    "B": {"position": [1, 0, 0]},
    "C": {"position": [0, 2, 0]},
    "D": {"position": [0, 0, 2]},
}

# 比较 dist(A,B) 和 dist(C,D)
constraint = compute_qrr(
    objects,
    pair1=("A", "B"),
    pair2=("C", "D"),
    metric=MetricType.DIST_3D,
    tau=0.10
)
print(constraint.comparator)  # "<" (A-B 距离小于 C-D)
```

### TRR 约束

```python
from ordinal_spatial.dsl.predicates import compute_trr

# 找到 A 相对于 B->C 轴的时钟位置
constraint = compute_trr(
    objects,
    target="A",
    ref1="B",
    ref2="C"
)
print(f"小时: {constraint.hour}, 象限: {constraint.quadrant}")
```

### 一致性检查

```python
from ordinal_spatial.evaluation.consistency import check_qrr_consistency

constraints = [
    {"pair1": ["A", "B"], "pair2": ["C", "D"], "comparator": "<"},
    {"pair1": ["C", "D"], "pair2": ["E", "F"], "comparator": "<"},
    {"pair1": ["E", "F"], "pair2": ["A", "B"], "comparator": "<"},  # 形成循环！
]

report = check_qrr_consistency(constraints)
print(f"一致: {report.is_consistent}")  # False
print(f"循环数: {len(report.cycles)}")  # 1
```

### 任务运行器

```python
from ordinal_spatial.tasks import T1QRRRunner, run_t1_qrr_evaluation
from ordinal_spatial.baselines.oracle import OracleBaseline

# 快速评估
baseline = OracleBaseline()
metrics = run_t1_qrr_evaluation(baseline, dataset, tau=0.10)
print(f"准确率: {metrics.accuracy}")

# 完整运行器配置
from ordinal_spatial.tasks.t1_classification import T1Config
config = T1Config(tau=0.10, save_predictions=True, output_dir="./results")
runner = T1QRRRunner(baseline, config)
result = runner.run(dataset)
```

## 项目结构

```
ordinal_spatial/
├── dsl/                    # 序关系约束语言
│   ├── comparators.py      # 基于容差的比较代数
│   ├── predicates.py       # QRR/TRR 实现
│   └── schema.py           # Pydantic 模型
│
├── evaluation/             # 指标与评估
│   ├── metrics.py          # 所有基准指标
│   └── consistency.py      # 约束图循环检测
│
├── generation/             # 场景与约束生成
│   ├── constraint_extractor.py
│   ├── difficulty_control.py
│   └── degeneracy_checker.py
│
├── baselines/              # 基线实现
│   ├── oracle.py           # 真值 oracle
│   ├── vlm_direct.py       # VLM 直接提示
│   ├── hybrid.py           # 预测-验证-修复循环
│   └── embedding.py        # 序嵌入优化
│
├── tasks/                  # 任务套件
│   ├── t1_classification.py
│   ├── t2_extraction.py
│   └── t3_reconstruction.py
│
├── agents/                 # VLM 约束提取智能体
│   ├── __init__.py         # 模块入口
│   ├── base.py             # 基类和数据结构
│   ├── vlm_constraint_agent.py    # VLM 智能体 (Task-2/3)
│   ├── blender_constraint_agent.py # Blender 智能体 (Task-1)
│   ├── cli.py              # 命令行接口
│   ├── prompts/            # 智能体提示词
│   │   └── constraint_extraction.py
│   └── tests/              # 智能体测试
│       └── test_vlm_agent.py
│
├── prompts/                # VLM 提示模板
│   ├── system_prompts.py
│   └── task_prompts.py
│
├── scripts/                # 命令行工具
│   ├── generate_dataset.py
│   ├── run_baseline.py
│   ├── evaluate.py
│   └── visualize.py
│
└── tests/                  # 单元测试
```

## 约束提取智能体

### 概述

本模块提供三种约束提取任务的智能体实现：

| 任务 | 智能体 | 描述 |
|------|--------|------|
| **Task-1** | `BlenderConstraintAgent` | 从 Blender 场景数据提取真值约束 |
| **Task-2** | `VLMConstraintAgent` | 从多视角图像提取约束 (VLM) |
| **Task-3** | `VLMConstraintAgent` | 从单视角图像提取约束 (VLM) |

### 命令行使用

```bash
# Task-3: 单视角约束提取
python -m ordinal_spatial.agents.cli extract \
    --image scene.png \
    --output constraints.json \
    --tau 0.10

# Task-2: 多视角约束提取
python -m ordinal_spatial.agents.cli extract \
    --images view1.png view2.png view3.png \
    --output constraints.json

# 使用自定义模型
python -m ordinal_spatial.agents.cli extract \
    --image scene.png \
    --model openai/gpt-4o \
    --output constraints.json
```

### API 使用

```python
# Task-1: Blender 真值提取
from ordinal_spatial.agents import BlenderConstraintAgent

agent = BlenderConstraintAgent()

# 从 CLEVR 场景文件提取
constraints_list = agent.extract_from_clevr_scenes(
    "output/CLEVR_scenes.json",
    tau=0.10
)

# 从 .blend 文件提取
constraints = agent.extract_from_blend_file(
    "scene.blend",
    tau=0.10
)

# Task-2/3: VLM 约束提取
from ordinal_spatial.agents import VLMConstraintAgent, VLMAgentConfig

config = VLMAgentConfig(
    model="google/gemma-3-27b-it",
    temperature=0.0,
)
agent = VLMConstraintAgent(config)

# 单视角提取 (Task-3)
result = agent.extract_from_single_view(
    "scene.png",
    tau=0.10,
)

# 多视角提取 (Task-2)
result = agent.extract_from_multi_view(
    ["view1.png", "view2.png", "view3.png"],
    tau=0.10,
)

# 输出结果
print(result.summary())
print(f"总约束数: {result.total_constraints()}")
```

### 输出格式 (QSP - 定性场景程序)

```json
{
  "objects": [
    {"id": "cube1", "type": "cube", "color": "red", "size_class": "large"}
  ],
  "constraints": {
    "axial": [{"obj1": "cube1", "obj2": "sphere1", "relation": "left_of"}],
    "topology": [{"obj1": "cube1", "obj2": "sphere1", "relation": "disjoint"}],
    "occlusion": [{"occluder": "cube1", "occluded": "sphere1", "partial": false}],
    "size": [{"bigger": "cube1", "smaller": "sphere1"}],
    "closer": [{"anchor": "cube1", "closer": "sphere1", "farther": "cone1"}],
    "qrr": [...],
    "trr": [...]
  },
  "confidence": 0.85
}
```

### 支持的约束类型

| 约束类型 | 描述 | 示例 |
|----------|------|------|
| **axial** | 二元轴向偏序 | left_of, right_of, above, below, in_front_of, behind |
| **topology** | RCC-8 拓扑关系 | disjoint, touching, overlapping |
| **occlusion** | 遮挡关系 | A 遮挡 B (视角相关) |
| **size** | 尺寸比较 | A 比 B 大 |
| **closer** | 三元距离比较 | B 比 C 更接近 A |
| **qrr** | 四元相对距离 | dist(A,B) < dist(C,D) |
| **trr** | 三元时钟方向 | A 在 B→C 轴的 3 点钟方向 |

## 容差参数 (tau)

容差参数 tau 控制比较的敏感度：

- `a <_tau b`: 当 `a < b * (1 - tau)` 时为真
- `a ~=_tau b`: 当 `|a - b| <= tau * max(a, b)` 时为真
- `a >_tau b`: 当 `a > b * (1 + tau)` 时为真

常用预设值：
- **严格 (strict)**: tau = 0.05
- **标准 (standard)**: tau = 0.10
- **宽松 (relaxed)**: tau = 0.20

## 难度等级

约束难度基于度量比率：

| 等级 | 比率范围 | 描述 |
|------|----------|------|
| 1 | > 2.0 | 简单 |
| 2 | 1.5 - 2.0 | 中等偏易 |
| 3 | 1.3 - 1.5 | 中等 |
| 4 | 1.15 - 1.3 | 中等偏难 |
| 5 | 1.05 - 1.15 | 困难 |
| 6 | 1.0 - 1.05 | 极难 |

## 基线模型

| 基线 | 描述 |
|------|------|
| **Oracle** | 从 3D 位置直接计算真值 |
| **VLM Direct** | 零样本 VLM 预测 |
| **VLM CoT** | 思维链提示 |
| **Hybrid** | 预测-验证-修复循环，带一致性检查 |
| **Embedding** | 梯度下降序嵌入 |

## 核心概念

### QRR (Quaternary Relative Relations) - 四元相对关系

QRR 比较两对物体之间的距离关系，形式为：`m(A,B) {<, ~=, >} m(C,D)`

其中 m 可以是：
- `DIST_3D`: 3D 欧氏距离
- `DIST_2D`: 2D 图像平面距离
- `DEPTH_GAP`: 深度差
- `SIZE_RATIO`: 尺寸比

### TRR (Ternary Clock Relations) - 三元时钟关系

TRR 描述目标物体相对于参考轴的方向，使用时钟表盘模型：
- 12 个小时位置（每个 30°）
- 4 个象限（每个 90°）
- 参考轴由两个物体定义

### 一致性检查

约束集必须全局一致，即不能存在矛盾循环：
- 如果 d(A,B) < d(C,D) 且 d(C,D) < d(E,F)，则 d(A,B) < d(E,F)
- 使用有向图循环检测算法验证

## 数据集划分

| 划分 | 样本数 | 用途 |
|------|--------|------|
| train | 10,000 | 训练 |
| val | 1,500 | 验证 |
| test_iid | 1,500 | 独立同分布测试 |
| test_comp | 500 | 组合泛化（更多物体） |
| test_view | 500 | 视角泛化（新视角） |
| test_hard | 500 | 困难样本（严格容差） |

## 测试

```bash
# 运行所有测试（70 个测试）
python -m pytest ordinal_spatial/tests/ -v

# 运行特定测试
python -m pytest ordinal_spatial/tests/test_dsl.py -v
python -m pytest ordinal_spatial/tests/test_consistency.py -v
```

## 示例工作流

### 1. 生成合成数据集

```python
from ordinal_spatial.scripts.generate_dataset import generate_random_scene

scene = generate_random_scene('scene_001', n_objects=8, tau=0.10)
print(f"生成了 {len(scene['constraints']['qrr'])} 个 QRR 约束")
print(f"生成了 {len(scene['constraints']['trr'])} 个 TRR 约束")
```

### 2. 评估 VLM 基线

```python
from ordinal_spatial.baselines.vlm_direct import VLMDirectBaseline, VLMConfig
from ordinal_spatial.tasks import T2Runner
from ordinal_spatial.tasks.t2_extraction import T2Config

# 配置 VLM
vlm_config = VLMConfig(model="openai/gpt-4o", api_key="your-key")
baseline = VLMDirectBaseline(vlm_config)

# 运行 T2 任务
task_config = T2Config(tau=0.10, check_consistency=True)
runner = T2Runner(baseline, task_config)
result = runner.run(dataset)

# 查看结果
print(f"精确率: {result.metrics.precision:.3f}")
print(f"召回率: {result.metrics.recall:.3f}")
print(f"F1 分数: {result.metrics.f1:.3f}")
print(f"一致性: {result.consistency_stats['n_consistent']} / {result.consistency_stats['n_consistent'] + result.consistency_stats['n_inconsistent']}")
```

### 3. 运行混合基线（带修复）

```python
from ordinal_spatial.baselines.hybrid import HybridBaseline, HybridConfig

# 配置混合基线
hybrid_config = HybridConfig(
    vlm_config=vlm_config,
    max_repair_iterations=3,
    require_consistency=True
)
baseline = HybridBaseline(hybrid_config)

# 提取约束（自动修复不一致）
result = baseline.extract_constraints(image, objects, tau=0.10)
print(f"迭代次数: {result['n_iterations']}")
print(f"最终一致: {result['final_consistent']}")
```

### 4. 序重构

```python
from ordinal_spatial.tasks import run_t3_evaluation
from ordinal_spatial.baselines.embedding import NumpyOrdinalEmbedding

# 使用 NumPy 嵌入器
embedder = NumpyOrdinalEmbedding(n_dims=3, max_iterations=1000)

# 运行 T3 评估
metrics = run_t3_evaluation(
    dataset,
    n_dims=3,
    max_iterations=1000
)
print(f"NRMS 误差: {metrics.nrms:.4f}")
print(f"约束满足率: {metrics.constraint_satisfaction_rate:.4f}")
```

## 故障分析

框架提供详细的故障模式分类：

- **F1**: 深度翻转（2D vs 3D 排序不匹配）
- **F2**: 透视错觉（前缩混淆）
- **F3**: 遮挡错误（隐藏物体误估）
- **F4**: 尺寸-距离混淆
- **F5**: 边界失败（接近 τ 阈值）
- **F6**: 传递性违反
- **F7**: 比较符翻转（< ↔ >）
- **F8**: 物体混淆
- **F9**: 度量混淆
- **F10**: 一致性崩溃
- **F11**: 约等号过度使用
- **F12**: 时钟漂移

## 许可证

MIT License

## 引用

如果您使用此基准测试，请引用：

```bibtex
@misc{ordinal-spatial-2024,
  title={ORDINAL-SPATIAL: A Benchmark for Evaluating Spatial Understanding in Vision-Language Models},
  author={},
  year={2024},
  url={https://github.com/yourusername/ordinal-spatial}
}
```

## 贡献

欢迎贡献！请提交 issue 或 pull request。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
