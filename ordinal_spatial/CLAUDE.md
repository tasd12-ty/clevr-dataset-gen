# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

Package is defined in the **parent** `pyproject.toml` (hatchling backend). Run all commands from the repo root (`clevr-dataset-gen/`):

```bash
pip install -e .            # core deps
pip install -e ".[dev]"     # all optional deps (ml, tokenizer, viz, test)
pip install -e ".[test]"    # just pytest
pip install -e ".[ml]"      # torch (for embedding baseline)
```

## Testing

Tests are configured via `[tool.pytest.ini_options]` in the parent `pyproject.toml`. Test paths: `ordinal_spatial/tests/` and `ordinal_spatial/agents/tests/`.

```bash
# all tests (run from repo root)
python -m pytest ordinal_spatial/tests/ ordinal_spatial/agents/tests/ -v

# single file
python -m pytest ordinal_spatial/tests/test_dsl.py -v

# single test
python -m pytest ordinal_spatial/tests/test_dsl.py::TestComparator::test_comparator_values -v

# with coverage
python -m pytest ordinal_spatial/tests/ --cov=ordinal_spatial --cov-report=html
```

There is no linter or formatter configured. No CI/CD pipeline.

## Architecture

The module evaluates VLM spatial reasoning through **ordinal constraints** — relative comparisons rather than absolute measurements.

### Core DSL (`dsl/`)

The foundation. Three files that define the constraint language:

- **`comparators.py`** — Tolerance-based three-way comparison algebra. `compare(a, b, tau)` returns `LT`/`APPROX`/`GT`. The tau parameter controls how close values must be to be considered approximately equal. `ComparatorChain` handles transitive reasoning.
- **`predicates.py`** — Two constraint types built on comparators:
  - **QRR** (Quaternary Relative Relations): "Is dist(A,B) < dist(C,D)?" — compares metrics across 4 objects
  - **TRR** (Ternary Clock Relations): "From target looking at ref1, where is ref2 on the clock face?" — directional relations using 12-hour clock
- **`schema.py`** — Pydantic v2 models for everything: `ObjectSpec`, `OrdinalSceneDescription`, `QRRConstraint`, `TRRConstraint`, queries, predictions. Use `.model_dump()` not `.dict()`.

### Evaluation (`evaluation/`)

- **`consistency.py`** — Graph-based cycle detection to verify constraint sets are self-consistent (QRR transitivity, TRR coherence)
- **`metrics.py`** — Per-task metrics: `T1QRRMetrics` (accuracy/F1), `T1TRRMetrics` (hour/quadrant accuracy), `T2Metrics` (precision/recall/F1/consistency), `T3Metrics` (NRMS/constraint satisfaction)
- **`constraint_diff.py`** — Fine-grained difference analysis between predicted and ground-truth constraints

### Four Evaluation Tasks

| Task | Module | What it tests |
|------|--------|---------------|
| T1-Q | `tasks/t1_classification.py` | QRR classification (given a constraint, predict LT/APPROX/GT) |
| T1-C | `tasks/t1_classification.py` | TRR classification (predict clock hour/quadrant) |
| T2 | `tasks/t2_extraction.py` | Extract full constraint set from images |
| T3 | `tasks/t3_reconstruction.py` | Reconstruct 3D layout from constraints |

### Baselines (`baselines/`)

- **Oracle** — Ground-truth 3D positions → perfect predictions
- **VLMDirect** — Zero-shot VLM queries with JSON output parsing
- **Hybrid** — Predict-verify-repair loop using consistency checker
- **OrdinalEmbedding** — Gradient-based optimization to find 3D positions satisfying constraints (T3 only, requires torch)

### Agents (`agents/`)

Higher-level extraction agents for constraint generation:

- **BlenderConstraintAgent** — Extracts ground-truth constraints from Blender scene files (Task-1)
- **VLMConstraintAgent** — Extracts constraints from images via VLM API (Task-2/3), supports single-view and multi-view

### Generation (`generation/`)

Dataset and constraint generation: `constraint_extractor.py` extracts all constraints from scenes, `degeneracy_checker.py` filters degenerate cases, `difficulty_control.py` manages difficulty levels.

### Scripts (`scripts/`)

CLI entry points run as `python -m ordinal_spatial.scripts.<name>`:

- `build_benchmark` — Full benchmark pipeline (Blender rendering + constraint extraction)
- `run_baseline` — Run a baseline on a task/split
- `generate_dataset` — Generate constraint datasets without rendering
- `validate_benchmark` — Validate dataset integrity
- `evaluate` — Compute metrics on predictions
- `visualize` — Plot results

## Code Conventions

- **2-space indentation**, 80-char line width
- **Chinese comments/docstrings**, English print/log/LLM-facing strings
- **snake_case** functions, **PascalCase** classes
- **Pydantic v2** for all data models
- **Lazy torch imports**: torch is optional — `baselines/__init__.py` imports `embedding.py` which uses torch. Class-level type hints must not reference `torch.Tensor` directly; use string forward references or guard with `TYPE_CHECKING`.

## Key Domain Concepts

- **tau (τ)**: Tolerance threshold for approximate equality. `a ≈ b` iff `|a-b| ≤ τ·max(a,b)`. Standard: 0.10, strict: 0.05, relaxed: 0.15.
- **QRR**: Compares `metric(A,B)` vs `metric(C,D)` where metric is one of: DIST_3D, DIST_2D, DEPTH_GAP, SIZE_RATIO.
- **TRR**: Given target T, ref1 R1, ref2 R2 — "standing at T facing R1, R2 is at N o'clock". Maps to 12 hours and 4 quadrants.
- **Difficulty levels** (1-6): Derived from how close the ratio is to 1.0. Level 1 = easy (ratio far from 1), level 6 = hardest (ratio ≈ 1).

---

## 完整任务流水线

整个项目的工作流为：**渲染场景 → 提取GT约束 → 生成数据集 → 运行基线/VLM → 评估 → 可视化**。

以下所有命令均从仓库根目录 `clevr-dataset-gen/` 运行。

### 第一步：数据生成

#### 方式A：完整Benchmark生成（渲染 + GT提取 + 划分）

需要 Blender 5.0+，一条命令完成：渲染场景图像 → 提取GT约束 → 切分train/val/test。

```bash
# tiny（测试用，约5场景/split）
python -m ordinal_spatial.scripts.build_benchmark \
    --output-dir ./data/benchmark_tiny \
    --blender-path /mnt/d/tools/blender/blender.exe \
    --tiny

# small（100/20/20/10/10 场景）
python -m ordinal_spatial.scripts.build_benchmark \
    --output-dir ./data/benchmark_small \
    --blender-path /mnt/d/tools/blender/blender.exe \
    --small

# 自定义规模
python -m ordinal_spatial.scripts.build_benchmark \
    --output-dir ./data/benchmark_full \
    --blender-path /mnt/d/tools/blender/blender.exe \
    --n-train 1000 --n-val 200 --n-test 500 \
    --n-test-comp 200 --n-test-hard 200

# GPU加速渲染
python -m ordinal_spatial.scripts.build_benchmark \
    --output-dir ./data/benchmark_full \
    --blender-path /mnt/d/tools/blender/blender.exe \
    --n-train 1000 --use-gpu
```

输出结构：每个场景生成 1 张单视角 + 4 张多视角图像（方位角 45°/135°/225°/315°），以及场景元数据 JSON（含GT约束）。

#### 方式B：8卡GPU并行生成（服务器场景）

```bash
# 交互模式
python scripts/build_8gpu.py --output ./data/full --size large --quality high

# 非交互模式（服务器上需加 --yes）
python scripts/build_8gpu.py --output ./data/full --size large --yes

# 规模预设：tiny(40场景) / small(1520) / medium(15200) / large(152000)
# 质量预设：draft / normal / high
```

#### 方式C：纯合成数据集（不渲染图像）

仅生成随机场景配置和对应约束，不调用 Blender 渲染。适合快速测试评估逻辑。

```bash
python -m ordinal_spatial.scripts.generate_dataset \
    --small --output-dir ./data/synthetic

python -m ordinal_spatial.scripts.generate_dataset \
    --n-train 10000 --n-val 1500 --n-test 1500 \
    --output-dir ./data/synthetic
```

### 第二步：数据验证

生成后检查数据完整性（目录结构、图像可读性、JSON合法性、约束一致性）。

```bash
python -m ordinal_spatial.scripts.validate_benchmark \
    --dataset-dir ./data/benchmark_tiny \
    --verbose

# 输出报告到文件
python -m ordinal_spatial.scripts.validate_benchmark \
    --dataset-dir ./data/benchmark_tiny \
    --output report.json
```

### 第三步：GT约束提取（Task-1）

Ground truth 通过 `BlenderConstraintAgent` 从3D场景坐标直接计算，无需VLM。

在 `build_benchmark` 流程中自动完成。也可单独调用：

```python
from ordinal_spatial.agents import BlenderConstraintAgent

agent = BlenderConstraintAgent()

# 从 CLEVR 场景JSON提取
constraints = agent.extract_from_clevr_scenes(
    "output/CLEVR_scenes.json", tau=0.10
)

# 从 .blend 文件提取
constraints = agent.extract_from_blend_file("scene.blend", tau=0.10)
```

提取的约束类型：QRR（距离比较）、TRR（时钟方向）、Topology（拓扑）、Occlusion（遮挡）、Axial（轴向）、Size（大小）、Closer（距离排序）。

### 第四步：VLM约束提取（Task-2/3）

通过 VLM API 从图像中提取空间约束。

```bash
# Task-3：单视角约束提取
python -m ordinal_spatial.agents.cli extract \
    --image scene.png \
    --model openai/gpt-4o \
    --tau 0.10 \
    --output constraints.json

# Task-2：多视角约束提取
python -m ordinal_spatial.agents.cli extract \
    --images view_0.png view_1.png view_2.png view_3.png \
    --model openai/gpt-4o \
    --output constraints.json

# 使用其他模型
python -m ordinal_spatial.agents.cli extract \
    --image scene.png \
    --model google/gemma-3-27b-it \
    --output constraints.json
```

需设置环境变量 `OPENROUTER_API_KEY`（使用 OpenRouter）或 `OPENAI_API_KEY`（使用 OpenAI）。

### 第五步：运行基线评估

基线方法在指定数据集和任务上生成预测结果。

```bash
# Oracle基线（GT直接计算，100%准确，用于上界参考）
python -m ordinal_spatial.scripts.run_baseline \
    --baseline oracle --task t1-q \
    --data ./data/benchmark_tiny --split test_iid \
    --output ./results/oracle

# VLM直接预测基线
python -m ordinal_spatial.scripts.run_baseline \
    --baseline vlm_direct --task t2 \
    --data ./data/benchmark_tiny \
    --model openai/gpt-4o \
    --output ./results/vlm_direct

# VLM + CoT（思维链）
python -m ordinal_spatial.scripts.run_baseline \
    --baseline vlm_cot --task t1-q \
    --data ./data/benchmark_tiny \
    --model openai/gpt-4o \
    --output ./results/vlm_cot

# Hybrid（预测-验证-修复循环）
python -m ordinal_spatial.scripts.run_baseline \
    --baseline hybrid --task t2 \
    --data ./data/benchmark_tiny \
    --model openai/gpt-4o \
    --output ./results/hybrid

# Embedding（梯度优化重构，仅T3任务，需torch）
python -m ordinal_spatial.scripts.run_baseline \
    --baseline embedding --task t3 \
    --data ./data/benchmark_tiny \
    --output ./results/embedding

# 所有任务一次跑完
python -m ordinal_spatial.scripts.run_baseline \
    --baseline oracle --task all \
    --data ./data/benchmark_tiny \
    --output ./results/oracle_all
```

可用基线：`oracle` / `vlm_direct` / `vlm_cot` / `hybrid` / `embedding`
可用任务：`t1-q` / `t1-c` / `t2` / `t3` / `all`
可用split：`train` / `val` / `test_iid` / `test_comp` / `test_hard`

### 第六步：评估预测结果

对预测结果计算各任务指标。

```bash
# T1-Q评估（准确率、宏F1、翻转率）
python -m ordinal_spatial.scripts.evaluate \
    --predictions ./results/oracle/t1_qrr/predictions.json \
    --ground-truth ./data/benchmark_tiny/ground_truth.json \
    --task t1-q \
    --output eval_t1q.json

# T1-C评估（小时准确率、象限准确率、角度误差）
python -m ordinal_spatial.scripts.evaluate \
    -p ./results/predictions.json \
    -g ./data/ground_truth.json \
    -t t1-c

# T2评估（精确率、召回率、F1、一致性率）
python -m ordinal_spatial.scripts.evaluate \
    -p preds.json -g gt.json -t t2

# T3评估（NRMS误差、约束满足率）
python -m ordinal_spatial.scripts.evaluate \
    -p preds.json -g gt.json -t t3

# 带失败分析
python -m ordinal_spatial.scripts.evaluate \
    -p preds.json -g gt.json -t t1-q --analyze-failures
```

### 第七步：3D重构（Task-3）

从约束集还原3D空间位置。

```bash
# 基本重构
python -m ordinal_spatial.reconstruction.cli \
    --input constraints.json --output ./reconstruction

# PyTorch求解器 + GPU
python -m ordinal_spatial.reconstruction.cli \
    --input constraints.json --output ./reconstruction \
    --solver pytorch --gpu --max-iterations 5000

# 带3D可视化
python -m ordinal_spatial.reconstruction.cli \
    --input constraints.json --output ./reconstruction \
    --visualizer plotly --show
```

### 第八步：可视化与对比

```bash
# 混淆矩阵（T1-Q）
python -m ordinal_spatial.scripts.visualize \
    --predictions preds.json --ground-truth gt.json \
    --task t1-q --output-dir ./plots

# 多基线对比
python -m ordinal_spatial.scripts.visualize \
    --compare ./results/oracle/summary.json \
              ./results/vlm_direct/summary.json \
              ./results/hybrid/summary.json \
    --output-dir ./comparison

# 全部图表
python -m ordinal_spatial.scripts.visualize \
    --predictions preds.json --ground-truth gt.json \
    --task t1-q --plot-type all --output-dir ./plots
```

图表类型：`confusion`（混淆矩阵）/ `difficulty`（难度分布）/ `consistency`（一致性）/ `reconstruction`（重构）/ `report`（完整报告）/ `all`

### 数据集Split说明

| Split | 场景数(默认) | 物体数 | tau | 用途 |
|-------|-------------|--------|-----|------|
| train | 1000 | 4-10 | 0.10 | 训练集 |
| val | 200 | 4-10 | 0.10 | 验证集 |
| test_iid | 200 | 4-10 | 0.10 | 同分布测试 |
| test_comp | 100 | 10-15 | 0.10 | 组合泛化（更多物体） |
| test_hard | 100 | 4-10 | 0.05 | 严格阈值（更难判断） |
