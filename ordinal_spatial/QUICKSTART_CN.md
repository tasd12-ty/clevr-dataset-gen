# ORDINAL-SPATIAL 快速入门指南

## 简介

ORDINAL-SPATIAL 是一个通过序关系约束评估视觉-语言模型（VLM）空间理解能力的基准测试框架。

## 核心概念

### 1. QRR（四元相对关系）

比较两对物体之间的距离：

```python
"dist(A,B) 与 dist(C,D) 的关系是什么？"
答案：< (小于) | ~= (约等于) | > (大于)
```

### 2. TRR（三元时钟关系）

使用时钟表盘描述方向：

```python
"A 相对于 B->C 轴的位置是几点钟？"
答案：1-12 小时 + 1-4 象限
```

### 3. 容差参数 τ (tau)

控制"约等于"的判定：

- `a <_τ b`: 当 `a < b × (1-τ)` 时
- `a ~=_τ b`: 当 `|a-b| ≤ τ × max(a,b)` 时
- `a >_τ b`: 当 `a > b × (1+τ)` 时

**默认值**：τ = 0.10 (10% 容差)

## 三大任务

### T1: 分类任务

**T1-Q: QRR 分类**
- 输入：图像 + 四个物体标识
- 输出：距离比较结果 (<, ~=, >)

**T1-C: TRR 分类**
- 输入：图像 + 三个物体标识
- 输出：时钟位置 (小时 + 象限)

### T2: 约束提取

- 输入：场景图像 + 物体列表
- 输出：完整的 QRR 和 TRR 约束集
- 评估：精确率、召回率、F1、一致性

### T3: 序重构

- 输入：约束集（无图像）
- 输出：满足约束的 3D 点配置
- 评估：NRMS 误差、约束满足率

## 5 分钟入门

### 1. 安装

```bash
cd ordinal_spatial
pip install -r requirements.txt
pip install -e .
```

### 2. 生成测试数据

```python
import numpy as np
from ordinal_spatial.scripts.generate_dataset import generate_random_scene

# 生成一个场景
np.random.seed(42)
scene = generate_random_scene('test_001', n_objects=5, tau=0.10)

print(f"物体数量: {len(scene['objects'])}")
print(f"QRR 约束: {len(scene['constraints']['qrr'])}")
print(f"TRR 约束: {len(scene['constraints']['trr'])}")
```

### 3. 使用 Oracle 基线（真值计算）

```python
from ordinal_spatial.baselines.oracle import OracleBaseline
from ordinal_spatial.dsl.schema import QRRQuery

# 创建 Oracle 基线
baseline = OracleBaseline()

# 构建物体字典
objects = {
    obj['id']: obj
    for obj in scene['objects']
}

# 创建查询
query = QRRQuery(
    scene_id=scene['scene_id'],
    query_id='q1',
    objects={'A': 'obj_0', 'B': 'obj_1', 'C': 'obj_2', 'D': 'obj_3'},
    metric='dist3D',
    tau=0.10
)

# 预测
result = baseline.predict_qrr(objects, query)
print(f"比较结果: {result.comparator}")
print(f"置信度: {result.confidence}")
```

### 4. 运行 T1-Q 评估

```python
from ordinal_spatial.tasks import run_t1_qrr_evaluation

# 准备数据集（简化示例）
dataset = [
    {
        'scene': scene,
        'qrr_queries': [
            {
                'query_id': 'q1',
                'A': 'obj_0',
                'B': 'obj_1',
                'C': 'obj_2',
                'D': 'obj_3',
                'metric': 'dist3D',
                'ground_truth': {'comparator': '<'}
            }
        ]
    }
]

# 评估
metrics = run_t1_qrr_evaluation(baseline, dataset, tau=0.10)
print(f"准确率: {metrics.accuracy:.3f}")
```

### 5. 一致性检查

```python
from ordinal_spatial.evaluation.consistency import check_qrr_consistency

# 构造约束集（包含矛盾）
constraints = [
    {'pair1': ['A', 'B'], 'pair2': ['C', 'D'], 'comparator': '<'},
    {'pair1': ['C', 'D'], 'pair2': ['E', 'F'], 'comparator': '<'},
    {'pair1': ['E', 'F'], 'pair2': ['A', 'B'], 'comparator': '<'},  # 矛盾！
]

# 检查一致性
report = check_qrr_consistency(constraints)
print(f"一致: {report.is_consistent}")  # False
print(f"循环数: {len(report.cycles)}")  # 1

if not report.is_consistent:
    for cycle in report.cycles:
        print(f"矛盾循环: {cycle}")
```

## 命令行工具

### 生成数据集

```bash
# 小型测试数据集
python -m ordinal_spatial.scripts.generate_dataset \
    --small \
    --output-dir ./data

# 自定义数据集
python -m ordinal_spatial.scripts.generate_dataset \
    --n-train 1000 \
    --n-val 150 \
    --n-test 150 \
    --min-objects 4 \
    --max-objects 12 \
    --tau 0.10 \
    --output-dir ./data
```

### 运行基线

```bash
# Oracle 基线
python -m ordinal_spatial.scripts.run_baseline \
    --baseline oracle \
    --task t1-q \
    --data ./data \
    --split test_iid \
    --output ./results

# VLM 基线（需要 API 密钥）
export OPENROUTER_API_KEY="sk-..."
python -m ordinal_spatial.scripts.run_baseline \
    --baseline vlm_direct \
    --task t2 \
    --data ./data \
    --model openai/gpt-4o \
    --output ./results
```

### 评估结果

```bash
python -m ordinal_spatial.scripts.evaluate \
    --predictions ./results/predictions.json \
    --ground-truth ./results/ground_truth.json \
    --task t1-q \
    --analyze-failures \
    --output ./results/metrics.json
```

### 可视化

```bash
python -m ordinal_spatial.scripts.visualize \
    --predictions ./results/predictions.json \
    --ground-truth ./results/ground_truth.json \
    --task t1-q \
    --output-dir ./visualizations
```

## 使用 VLM 基线

```python
from ordinal_spatial.baselines.vlm_direct import VLMDirectBaseline, VLMConfig
from ordinal_spatial.tasks import T2Runner
from ordinal_spatial.tasks.t2_extraction import T2Config

# 配置 VLM
vlm_config = VLMConfig(
    model="openai/gpt-4o",
    api_key="your-api-key",
    temperature=0.0,
    with_cot=True  # 启用思维链
)
baseline = VLMDirectBaseline(vlm_config)

# 配置任务
task_config = T2Config(
    tau=0.10,
    check_consistency=True,
    save_predictions=True,
    output_dir="./results"
)

# 运行评估
runner = T2Runner(baseline, task_config)
result = runner.run(dataset, images_dir="./images")

# 查看结果
print(f"精确率: {result.metrics.precision:.3f}")
print(f"召回率: {result.metrics.recall:.3f}")
print(f"F1 分数: {result.metrics.f1:.3f}")
print(f"一致: {result.consistency_stats['n_consistent']}")
print(f"不一致: {result.consistency_stats['n_inconsistent']}")
```

## 混合基线（带修复）

```python
from ordinal_spatial.baselines.hybrid import HybridBaseline, HybridConfig

# 配置混合基线
hybrid_config = HybridConfig(
    vlm_config=vlm_config,
    max_repair_iterations=3,  # 最多修复 3 次
    require_consistency=True
)
baseline = HybridBaseline(hybrid_config)

# 提取约束（自动修复不一致）
result = baseline.extract_constraints(
    image="scene.png",
    objects=objects,
    tau=0.10
)

print(f"迭代次数: {result['n_iterations']}")
print(f"最终一致: {result['final_consistent']}")
print(f"约束数: {len(result['qrr'])}")
```

## 故障模式

框架自动识别常见错误类型：

| 代码 | 故障类型 | 描述 |
|------|---------|------|
| F1 | 深度翻转 | 2D vs 3D 排序不匹配 |
| F2 | 透视错觉 | 前缩混淆 |
| F3 | 遮挡错误 | 隐藏物体误估 |
| F5 | 边界失败 | 接近 τ 阈值的错误 |
| F6 | 传递性违反 | 逻辑不一致 |
| F7 | 比较符翻转 | < ↔ > 系统性错误 |
| F10 | 一致性崩溃 | 多个矛盾 |

## 测试

```bash
# 运行所有测试
python -m pytest ordinal_spatial/tests/ -v

# 运行特定测试
python -m pytest ordinal_spatial/tests/test_dsl.py -v
python -m pytest ordinal_spatial/tests/test_consistency.py -v

# 查看覆盖率
python -m pytest --cov=ordinal_spatial --cov-report=html
```

## 常见问题

### Q: 如何选择 τ 值？

**A:**
- τ = 0.05: 严格模式，约 10% 的比较为"约等于"
- τ = 0.10: 标准模式（推荐），约 19% 约等于
- τ = 0.20: 宽松模式，约 33% 约等于

### Q: VLM 返回的不是 JSON 怎么办？

**A:** VLMDirectBaseline 会自动尝试从 markdown 代码块中提取 JSON：
```python
# 自动处理 ```json ... ``` 包裹的响应
```

### Q: 如何处理不一致的约束？

**A:** 使用 HybridBaseline，它会自动检测并修复：
```python
baseline = HybridBaseline(config)
result = baseline.extract_constraints(image, objects, tau=0.10)
# 自动进行预测-验证-修复循环
```

### Q: 如何可视化约束图？

**A:** 使用 NetworkX 和 matplotlib：
```python
from ordinal_spatial.evaluation.consistency import build_constraint_graph
import networkx as nx
import matplotlib.pyplot as plt

G = build_constraint_graph(constraints)
nx.draw(G, with_labels=True)
plt.show()
```

## 下一步

1. 阅读完整的 [README.md](README.md)
2. 查看 [测试示例](tests/)
3. 探索 [提示词模板](prompts/)
4. 运行完整的评估流程

## 获取帮助

- GitHub Issues: 报告 bug 或请求功能
- 文档: 查看 `ordinal_spatial/` 下的 docstrings
- 示例: 参考 `tests/` 中的测试用例

## 许可证

MIT License - 自由使用和修改
