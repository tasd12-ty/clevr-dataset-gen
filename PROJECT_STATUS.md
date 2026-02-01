# VLM 三维空间理解与"逆向重绘"基准 - 项目状态报告

> 基于 `3D_redraw.pdf` 文档目标的实现状态分析
> 更新日期: 2026-02-01

---

## 一、项目目标概述

### 核心理念

- **不要求**模型从单张图像恢复精确的相机位姿与绝对坐标
- **要求**模型恢复并保持"人类可感知的相对空间关系/约束集合"
- 通过"重绘 (Re-drawing)"来验证理解的完整性

### 四大核心任务

| 任务 | 描述 | 输入 | 输出 |
|------|------|------|------|
| **Task-1** | Code → Constraints (GT 上界) | `.blend` 文件 / Python 脚本 | C_code = ⟨O, R_GT⟩ |
| **Task-2** | Multi-view → Constraints (感知上界) | 多视角图像 {I_1, ..., I_K} | C_multi |
| **Task-3** | Single-view → Constraints (核心挑战) | 单张 RGB 图像 | C_single |
| **Task-4** | Constraints → Reconstruction | 约束集合 | 3D 场景重建 |

---

## 二、实现状态总览

### 2.1 当前实现状态

| 任务 | Method | Implementation | 状态 |
|------|--------|----------------|------|
| Task-1 | ✅ | ✅ | **已完成** |
| Task-2 | ✅ | ✅ | **已完成** |
| Task-3 | ✅ | ✅ | **已完成** |
| Task-4 | ✅ | ✅ | **已完成** |

### 2.2 缺失部分

| 缺失项 | 类型 |
|--------|------|
| Constraint-Diff 完整指标 | 评估 |
| 多视角图像渲染脚本 | 数据 |
| Benchmark 图像数据集 | 数据 |
| 端到端实验流程脚本 | 实验 |

---

## 三、模块稳定性分析（论文实验视角）

### 3.1 稳定性分级

```
██████████ 极稳定 - 一旦定义基本不变
████████░░ 较稳定 - 框架固定，细节可能微调
██████░░░░ 中等   - 核心逻辑稳定，参数/策略可能调整
████░░░░░░ 易变   - 根据实验效果需要迭代
██░░░░░░░░ 高变   - 频繁调整，依赖实验反馈
```

### 3.2 各模块稳定性评估

| 模块 | 稳定性 | 说明 | 依赖 |
|------|--------|------|------|
| **Blender 场景渲染** | ██████████ | 基于成熟的 CLEVR 流程 | 无 |
| **DSL Schema 定义** | ██████████ | 已完成，语言规范固定 | 无 |
| **Task-1 GT 提取** | ██████████ | 纯几何计算，确定性 | DSL |
| **单视角图像渲染** | ██████████ | 标准渲染流程 | Blender |
| **多视角图像渲染** | ████████░░ | 相机轨道参数可能微调 | Blender |
| **Constraint-Diff 指标** | ████████░░ | 定义明确，计算逻辑固定 | DSL |
| **数据集划分策略** | ████████░░ | train/val/test 标准流程 | 渲染 |
| **Task-4 求解器框架** | ██████░░░░ | 框架稳定，损失权重需调 | DSL |
| **Task-2 多视角融合** | ██████░░░░ | 策略可能根据实验调整 | VLM |
| **实验运行脚本** | ████░░░░░░ | 依赖其他模块接口 | 全部 |
| **VLM 提示词** | ██░░░░░░░░ | 需根据模型效果迭代 | 无 |
| **VLM 模型选择** | ██░░░░░░░░ | 对比实验可能换模型 | 无 |

### 3.3 依赖关系图

```
                    ┌─────────────────┐
                    │  DSL Schema     │ (极稳定)
                    │  形式化语言定义  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐  ┌─────────────────┐  ┌─────────────┐
    │ Task-1      │  │ Constraint-Diff │  │ Task-4      │
    │ GT 提取     │  │ 评估指标        │  │ 求解器      │
    │ (极稳定)    │  │ (较稳定)        │  │ (中等)      │
    └──────┬──────┘  └────────┬────────┘  └─────────────┘
           │                  │
           ▼                  │
    ┌─────────────────┐       │
    │ Blender 渲染    │       │
    │ 单视角+多视角   │       │
    │ (极稳定/较稳定) │       │
    └────────┬────────┘       │
             │                │
             ▼                │
    ┌─────────────────┐       │
    │ Benchmark       │       │
    │ 图像数据集      │       │
    │ (较稳定)        │       │
    └────────┬────────┘       │
             │                │
             ▼                ▼
    ┌─────────────────────────────────┐
    │      VLM 实验 (Task-2/3)        │
    │  提示词 + 模型选择 (高变)        │
    └────────────────┬────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────┐
    │      端到端实验脚本              │
    │      结果分析与报告              │
    └─────────────────────────────────┘
```

---

## 四、推荐实现顺序

### 原则

1. **先稳定后易变** - 避免底层变动导致上层重写
2. **先数据后实验** - 数据集是所有实验的基础
3. **先评估后调优** - 有指标才能迭代改进
4. **VLM 相关最后做** - 最易变，放最后迭代

### 推荐顺序

```
Phase 1: 数据基础设施 (极稳定)
├── 1.1 多视角渲染脚本
├── 1.2 Benchmark 图像数据集生成
└── 1.3 数据集验证与统计

Phase 2: 评估框架 (较稳定)
├── 2.1 Constraint-Diff 指标实现
└── 2.2 评估报告生成模板

Phase 3: VLM 实验 (易变)
├── 3.1 端到端实验脚本框架
├── 3.2 VLM 提示词迭代
├── 3.3 多模型对比实验
└── 3.4 Task-2 多视角融合策略
```

---

## 五、详细任务分解

### Phase 1: 数据基础设施

#### 1.1 多视角渲染脚本 ⭐ 最优先

**稳定性**: ████████░░ (较稳定)

**为什么先做**:
- 基于成熟的 CLEVR 渲染流程，变动风险低
- 是生成 Benchmark 数据集的前置条件
- 与 VLM 代码完全解耦，不受后续迭代影响

**实现要点**:
```python
# 需要创建: image_generation/render_multiview.py

class MultiViewRenderer:
    def __init__(self, n_views=4, camera_distance=15.0):
        self.n_views = n_views
        self.camera_distance = camera_distance

    def render_scene(self, scene_file, output_dir):
        """渲染单个场景的多视角图像"""
        # 相机绕 Z 轴均匀分布
        # 输出: view_0.png, view_1.png, ...
```

**相机配置建议** (可固定):
- 视角数量: 4 (正交视角，覆盖 0°/90°/180°/270°)
- 相机距离: 15 单位 (与 CLEVR 默认一致)
- 仰角: 30° (固定)

#### 1.2 Benchmark 图像数据集生成

**稳定性**: ████████░░ (较稳定)

**为什么紧接着做**:
- 依赖 1.1 的渲染脚本
- 数据集一旦生成，可反复用于所有实验
- 不依赖任何 VLM 代码

**数据规模建议**:
```
splits/
├── train.json      (1000 场景)
├── val.json        (200 场景)
├── test_iid.json   (200 场景)
├── test_comp.json  (100 场景, 更多物体)
└── test_hard.json  (100 场景, 小 tau)

images/
├── single_view/    (~1600 张)
└── multi_view/     (~1600 × 4 = 6400 张)
```

#### 1.3 数据集验证

**检查项**:
- [ ] 所有图像正常渲染
- [ ] GT 约束与图像对应
- [ ] 无退化场景（物体重叠等）
- [ ] 数据统计报告

---

### Phase 2: 评估框架

#### 2.1 Constraint-Diff 指标实现

**稳定性**: ████████░░ (较稳定)

**为什么第二阶段做**:
- 定义已在 PDF 中明确，不会大改
- 是评估所有实验的基础
- 与 VLM 代码解耦

**需要修改**: `ordinal_spatial/evaluation/metrics.py`

```python
@dataclass
class ConstraintDiffMetrics:
    """PDF 第5-6页定义的 Constraint-Diff 指标"""

    # 绝对数量
    n_ground_truth: int      # |R_GT|
    n_predicted: int         # |R_pred|
    n_correct: int           # |R_GT ∩ R_pred| 且值相同
    n_missing: int           # |R_GT \ R_pred|
    n_spurious: int          # |R_pred \ R_GT|
    n_violated: int          # 存在但方向反转

    # 比率
    missing_rate: float      # n_missing / n_ground_truth
    spurious_rate: float     # n_spurious / n_predicted
    violated_rate: float     # n_violated / (n_correct + n_violated)

    # 兼容原有指标
    precision: float         # n_correct / n_predicted
    recall: float            # n_correct / n_ground_truth
    f1: float

def compute_constraint_diff(
    predicted: ConstraintSet,
    ground_truth: ConstraintSet
) -> ConstraintDiffMetrics:
    """计算约束差分指标"""
    ...
```

#### 2.2 评估报告模板

**输出格式**:
```json
{
  "model": "gpt-4o",
  "task": "t3_single_view",
  "dataset": "test_iid",
  "metrics": {
    "constraint_diff": {
      "missing_rate": 0.15,
      "spurious_rate": 0.08,
      "violated_rate": 0.03
    },
    "f1": 0.82,
    "self_consistency": 0.95
  },
  "per_scene_results": [...]
}
```

---

### Phase 3: VLM 实验

#### 3.1 端到端实验脚本框架

**稳定性**: ████░░░░░░ (易变)

**为什么最后做**:
- 依赖 Phase 1/2 的所有组件
- 接口可能随 VLM 提示词调整而变化
- 需要根据实验反馈迭代

**脚本结构**:
```python
# scripts/run_vlm_experiment.py

def main():
    # 1. 加载数据集
    # 2. 初始化 VLM Agent
    # 3. 批量推理
    # 4. 计算 Constraint-Diff
    # 5. 生成报告
```

#### 3.2 VLM 提示词迭代

**稳定性**: ██░░░░░░░░ (高变)

**预期迭代点**:
- 约束类型的描述方式
- Few-shot 示例的选择
- 输出格式的约束强度
- Chain-of-Thought 策略

#### 3.3 多模型对比

**候选模型**:
- GPT-4o / GPT-4V
- Claude 3.5 Sonnet
- Gemini 1.5 Pro
- Gemma 3 27B (当前默认)

#### 3.4 Task-2 多视角融合策略

**稳定性**: ██████░░░░ (中等)

**可能的策略**:
- 取交集（当前实现，保守）
- 加权投票
- 一致性过滤

---

## 六、风险与注意事项

### 避免的陷阱

1. **不要先优化 VLM 提示词**
   - 没有数据集无法评估效果
   - 提示词必然需要多轮迭代

2. **不要过早固定实验脚本接口**
   - 等 Phase 1/2 完成后再设计
   - 预留扩展点

3. **不要跳过数据验证**
   - 错误的 GT 会导致所有实验结论错误
   - 花时间验证是值得的

### 检查点

| 阶段完成后 | 验证项 |
|-----------|--------|
| Phase 1 | 随机抽查 10 个场景，人工确认图像和 GT 一致 |
| Phase 2 | 用 Oracle baseline 跑通完整流程，指标应接近满分 |
| Phase 3 | 至少 2 个模型完成对比实验 |

---

## 七、文件索引

### 已实现

| 模块 | 文件 |
|------|------|
| VLM Agent | `ordinal_spatial/agents/vlm_constraint_agent.py` |
| Blender Agent | `ordinal_spatial/agents/blender_constraint_agent.py` |
| DSL | `ordinal_spatial/dsl/schema.py` |
| 评估指标 | `ordinal_spatial/evaluation/metrics.py` |
| 约束求解 | `ordinal_spatial/reconstruction/constraint_solver.py` |
| 场景生成 | `ordinal_spatial/scripts/generate_dataset.py` |
| 图像渲染 | `image_generation/render_images.py` |

### 待实现

| 模块 | 建议文件 |
|------|----------|
| 多视角渲染 | `image_generation/render_multiview.py` |
| Constraint-Diff | `ordinal_spatial/evaluation/constraint_diff.py` |
| 实验脚本 | `ordinal_spatial/scripts/run_vlm_experiment.py` |
| 数据集构建 | `ordinal_spatial/scripts/build_benchmark.py` |
