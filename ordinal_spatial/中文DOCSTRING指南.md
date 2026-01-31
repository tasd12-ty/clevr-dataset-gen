# 中文 Docstring 添加指南

## 当前状态

### 已完成的函数（5个）

#### dsl/comparators.py
- ✅ `compare()` - 基于容差的比较（核心函数）
- ✅ `compare_ratio()` - 比较并返回比率
- ✅ `ordinal_distance()` - 计算序距离
- ✅ `is_flip()` - 检查是否翻转
- ✅ `difficulty_from_ratio()` - 从比率确定难度

### 待完成统计

| 模块 | 函数/类数量 | 优先级 |
|------|------------|--------|
| dsl/predicates.py | 27 | ⭐⭐⭐ 高（核心 QRR/TRR） |
| dsl/schema.py | 31 | ⭐⭐ 中（数据模型） |
| evaluation/metrics.py | 21 | ⭐⭐⭐ 高（评估指标） |
| evaluation/consistency.py | 15 | ⭐⭐ 中（一致性检查） |
| baselines/oracle.py | 11 | ⭐⭐⭐ 高（基线接口） |
| baselines/vlm_direct.py | 15 | ⭐⭐ 中（VLM基线） |
| tasks/t1_classification.py | 12 | ⭐⭐⭐ 高（任务运行器） |
| tasks/t2_extraction.py | 8 | ⭐⭐ 中（任务运行器） |
| tasks/t3_reconstruction.py | 10 | ⭐⭐ 中（任务运行器） |

**总计**: 192 个函数/类

## 标准格式

### 模板1: 基本函数

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    中文简短描述（一句话）。

    详细说明（可选）：
    - 要点1
    - 要点2

    参数:
        param1: 参数1的中文说明
        param2: 参数2的中文说明

    返回:
        返回值的中文说明

    示例:
        >>> code_example()
        expected_result

    Raises:
        ExceptionType: 异常说明（如果有）

    English short description.

    Detailed explanation (optional):
    - Point 1
    - Point 2

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Examples:
        >>> code_example()
        expected_result

    Raises:
        ExceptionType: Exception description (if any)
    """
    pass
```

### 模板2: 类

```python
class ClassName:
    """
    类的中文简短描述。

    详细说明：
    - 用途说明
    - 主要特性

    属性:
        attr1: 属性1说明
        attr2: 属性2说明

    示例:
        >>> obj = ClassName()
        >>> obj.method()

    English short description of the class.

    Detailed explanation:
    - Purpose
    - Main features

    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2

    Examples:
        >>> obj = ClassName()
        >>> obj.method()
    """
    pass
```

## 优先级列表（推荐添加顺序）

### 第一优先级（最常用的公开 API）

#### dsl/predicates.py
1. `compute_qrr()` - 计算 QRR 约束（核心）
2. `compute_trr()` - 计算 TRR 约束（核心）
3. `extract_all_qrr()` - 提取所有 QRR 约束
4. `extract_all_trr()` - 提取所有 TRR 约束
5. `QRRConstraint` 类 - QRR 数据结构
6. `TRRConstraint` 类 - TRR 数据结构

#### evaluation/metrics.py
1. `compute_t1_qrr_metrics()` - 计算 T1-Q 指标
2. `compute_t1_trr_metrics()` - 计算 T1-C 指标
3. `compute_t2_metrics()` - 计算 T2 指标
4. `compute_t3_metrics()` - 计算 T3 指标
5. `T1QRRMetrics` 类 - T1-Q 指标数据类
6. `T2Metrics` 类 - T2 指标数据类

#### baselines/oracle.py
1. `OracleBaseline` 类 - Oracle 基线
2. `predict_qrr()` - QRR 预测方法
3. `predict_trr()` - TRR 预测方法

#### tasks/t1_classification.py
1. `T1QRRRunner` 类 - T1-Q 运行器
2. `T1TRRRunner` 类 - T1-C 运行器
3. `run_t1_qrr_evaluation()` - 快速评估函数
4. `run_t1_trr_evaluation()` - 快速评估函数

### 第二优先级（辅助函数）

#### dsl/comparators.py
- `ComparatorChain` 类 - 比较器链
- 类方法（`flip()`, `from_string()`等）

#### evaluation/consistency.py
- `check_qrr_consistency()` - 检查 QRR 一致性
- `check_trr_consistency()` - 检查 TRR 一致性
- `ConsistencyChecker` 类 - 一致性检查器

#### baselines/vlm_direct.py
- `VLMDirectBaseline` 类 - VLM 直接基线
- `predict_qrr()` 方法
- `extract_constraints()` 方法

### 第三优先级（内部辅助）

- dsl/schema.py 中的数据模型类
- generation 模块中的辅助函数
- 私有方法和内部工具函数

## 批量添加工具

### 方法1: 使用脚本模板

```bash
# 为单个文件添加中文 docstring
python scripts/add_chinese_docstrings.py --file dsl/predicates.py

# 为整个模块添加
python scripts/add_chinese_docstrings.py --module dsl
```

### 方法2: 手动添加（推荐）

对于核心函数，推荐手动添加以确保翻译质量：

1. 打开文件
2. 找到函数定义
3. 在现有英文 docstring 前添加中文部分
4. 保持缩进和格式一致
5. 测试代码仍能正常运行

## 翻译术语对照表

| 英文 | 中文 | 备注 |
|------|------|------|
| Compare | 比较 | |
| Compute | 计算 | |
| Extract | 提取 | |
| Evaluate | 评估 | |
| Constraint | 约束 | |
| Tolerance | 容差 | |
| Comparator | 比较器 | |
| Metric | 度量 | |
| Distance | 距离 | |
| Ratio | 比率 | |
| Difficulty | 难度 | |
| Prediction | 预测 | |
| Ground truth | 真值 | |
| Accuracy | 准确率 | |
| Precision | 精确率 | |
| Recall | 召回率 | |
| Consistency | 一致性 | |
| Threshold | 阈值 | |
| Baseline | 基线 | |
| Runner | 运行器 | |
| Query | 查询 | |
| Scene | 场景 | |
| Object | 物体 | |
| Position | 位置 | |
| Ordinal | 序/序关系 | |
| Quaternary | 四元 | |
| Ternary | 三元 | |
| Clock | 时钟 | |
| Quadrant | 象限 | |
| Reconstruction | 重构 | |
| Embedding | 嵌入 | |
| Configuration | 配置 | |

## 检查清单

添加中文 docstring 时请确保：

- [ ] 中文部分在英文之前
- [ ] 保持原有缩进
- [ ] 参数说明完整（Args/参数）
- [ ] 返回值说明清晰（Returns/返回）
- [ ] 示例代码正确（Examples/示例）
- [ ] 异常说明完整（Raises，如果有）
- [ ] 空行分隔中英文部分
- [ ] 专业术语翻译准确
- [ ] 测试代码仍能运行

## 实施建议

### 阶段1：核心 API（建议优先完成）

花费时间：约 2-3 小时
函数数量：约 20-30 个
覆盖范围：80% 的常用功能

重点文件：
- `dsl/predicates.py`（6个核心函数）
- `evaluation/metrics.py`（6个计算函数）
- `baselines/oracle.py`（3个主要方法）
- `tasks/t1_classification.py`（4个运行函数）

### 阶段2：辅助功能（可选）

花费时间：约 3-4 小时
函数数量：约 40-50 个
覆盖范围：剩余 20% 功能

### 阶段3：完整覆盖（未来可做）

花费时间：约 5-6 小时
函数数量：约 100+ 个
包括所有私有方法和工具函数

## 质量保证

### 自动检查

```python
# 检查文件中是否有未翻译的 docstring
python scripts/check_chinese_docstrings.py ordinal_spatial/

# 输出：
# ✓ dsl/comparators.py: 8/8 函数有中文 docstring
# ⚠ dsl/predicates.py: 6/27 函数有中文 docstring
# ...
```

### 手动审核

1. 阅读中文 docstring，确保：
   - 语句通顺
   - 术语准确
   - 格式正确
   - 与英文对应

2. 运行测试确保代码未破坏：
   ```bash
   python -m pytest ordinal_spatial/tests/ -v
   ```

3. 测试 help() 函数：
   ```python
   from ordinal_spatial.dsl.comparators import compare
   help(compare)
   # 应该看到中英文双语文档
   ```

## 维护建议

1. **新函数添加时**：立即添加中英文 docstring
2. **API 修改时**：同步更新中英文说明
3. **定期检查**：使用自动化工具检查覆盖率
4. **代码审查**：PR 时检查 docstring 质量

## 联系方式

如果需要帮助或有疑问：
- 查看本指南的模板和示例
- 参考已完成的函数（如 `compare()`）
- 使用提供的术语对照表

---

**文档版本**: v1.0
**最后更新**: 2024年
**维护者**: 项目团队
