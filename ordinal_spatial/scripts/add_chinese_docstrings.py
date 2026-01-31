#!/usr/bin/env python3
"""
自动为 Python 文件中的函数和类添加中文 docstring。

此脚本会：
1. 扫描所有 Python 文件
2. 识别函数和类的 docstring
3. 为英文 docstring 添加中文翻译
4. 保持原有格式和缩进
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

# 中英文对照词典
TRANSLATION_MAP = {
    # 常用动词
    "Compare": "比较",
    "Calculate": "计算",
    "Compute": "计算",
    "Check": "检查",
    "Determine": "确定",
    "Extract": "提取",
    "Generate": "生成",
    "Evaluate": "评估",
    "Validate": "验证",
    "Build": "构建",
    "Create": "创建",
    "Initialize": "初始化",
    "Load": "加载",
    "Save": "保存",
    "Format": "格式化",
    "Parse": "解析",
    "Run": "运行",
    "Fit": "拟合",

    # 常用名词
    "Args": "参数",
    "Returns": "返回",
    "Examples": "示例",
    "Attributes": "属性",
    "Parameters": "参数",
    "Note": "注意",
    "Warning": "警告",

    # 技术术语
    "tolerance": "容差",
    "comparator": "比较器",
    "constraint": "约束",
    "metric": "度量",
    "distance": "距离",
    "ratio": "比率",
    "difficulty": "难度",
    "prediction": "预测",
    "ground truth": "真值",
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "consistency": "一致性",
    "threshold": "阈值",
}


def translate_simple(text: str) -> str:
    """
    简单翻译英文文本为中文。

    这是一个非常基础的翻译函数，仅用于演示。
    实际使用中应该用更好的翻译服务。

    参数:
        text: 英文文本

    返回:
        中文翻译（如果无法翻译则返回原文）
    """
    # 这里只是示例，实际应该使用专业翻译 API
    result = text
    for en, zh in TRANSLATION_MAP.items():
        result = result.replace(en, zh)
    return result


def extract_docstring(node: ast.FunctionDef) -> str:
    """
    从 AST 节点提取 docstring。

    参数:
        node: AST 函数定义节点

    返回:
        docstring 文本（如果有）
    """
    return ast.get_docstring(node) or ""


def create_bilingual_docstring(original: str) -> str:
    """
    创建中英文双语 docstring。

    将英文 docstring 转换为中英文双语格式：
    - 中文翻译在前
    - 空行分隔
    - 英文原文在后

    参数:
        original: 原始英文 docstring

    返回:
        中英文双语 docstring
    """
    if not original or original.strip() == "":
        return original

    # 简单示例：为演示添加中文标记
    # 实际使用中需要调用翻译 API
    chinese_part = f"[中文翻译]\n{translate_simple(original)}\n\n"
    english_part = f"[English Original]\n{original}"

    return chinese_part + english_part


def process_file(filepath: Path, dry_run: bool = True):
    """
    处理单个 Python 文件。

    参数:
        filepath: Python 文件路径
        dry_run: 如果为 True，只打印预览不实际修改
    """
    print(f"\n处理文件 | Processing: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        # 统计函数和类
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        print(f"  发现 {len(functions)} 个函数，{len(classes)} 个类")

        if dry_run:
            # 只显示前几个函数的 docstring
            for i, func in enumerate(functions[:3]):
                docstring = extract_docstring(func)
                if docstring:
                    print(f"\n  函数: {func.name}")
                    print(f"  原 docstring: {docstring[:100]}...")

    except Exception as e:
        print(f"  错误: {e}")


def main():
    """主函数"""
    print("="*60)
    print("自动添加中文 docstring 工具")
    print("="*60)

    base_dir = Path(__file__).parent.parent

    # 扫描所有 Python 文件
    python_files = []
    for pattern in ["dsl/*.py", "evaluation/*.py", "generation/*.py",
                    "baselines/*.py", "tasks/*.py", "prompts/*.py"]:
        python_files.extend(base_dir.glob(pattern))

    print(f"\n找到 {len(python_files)} 个 Python 文件")

    # 处理每个文件（dry run 模式）
    for filepath in python_files[:3]:  # 只处理前3个文件作为示例
        process_file(filepath, dry_run=True)

    print("\n" + "="*60)
    print("注意：这是 dry-run 模式")
    print("实际使用时需要：")
    print("1. 集成专业翻译 API（如 DeepL, Google Translate）")
    print("2. 手动审核翻译质量")
    print("3. 保持代码格式和缩进")
    print("="*60)


if __name__ == "__main__":
    main()
