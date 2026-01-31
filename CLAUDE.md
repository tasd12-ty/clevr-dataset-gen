# CLAUDE.md - AI Assistant Guide for CLEVR Dataset Generation

## Repository Overview

This repository contains code for generating the **CLEVR (Compositional Language and Elementary Visual Reasoning) dataset** and the **ORDINAL-SPATIAL spatial reasoning benchmark**. It provides tools for:

1. Rendering synthetic 3D scenes using Blender
2. Generating compositional natural language questions with functional programs
3. Evaluating Vision-Language Models (VLMs) on ordinal spatial reasoning tasks

## Directory Structure

```
clevr-dataset-gen/
├── image_generation/          # Blender-based image rendering
│   ├── render_images.py       # Main rendering script
│   ├── collect_scenes.py      # Scene collection utility
│   ├── utils.py               # Blender utilities
│   └── data/                  # Blender assets (shapes, materials)
│
├── question_generation/       # Question synthesis
│   ├── generate_questions.py  # Main question generation script
│   ├── question_engine.py     # Template processing engine
│   ├── metadata.json          # Functional programming language specs
│   ├── synonyms.json          # Natural language synonyms
│   └── CLEVR_1.0_templates/   # Question templates (~40 files)
│
├── ordinal_spatial/           # Spatial reasoning benchmark module
│   ├── dsl/                   # Domain-Specific Language
│   │   ├── schema.py          # Pydantic models (ObjectSpec, OSD)
│   │   ├── comparators.py     # Tolerance algebra
│   │   └── predicates.py      # QRR/TRR computation
│   ├── evaluation/            # Metrics and consistency checking
│   │   ├── metrics.py         # T1/T2/T3 evaluation metrics
│   │   └── consistency.py     # Constraint graph validation
│   ├── generation/            # Constraint and dataset generation
│   │   ├── constraint_extractor.py
│   │   ├── degeneracy_checker.py
│   │   └── difficulty_control.py
│   ├── baselines/             # Baseline implementations
│   │   ├── oracle.py          # Ground truth baseline
│   │   ├── vlm_direct.py      # VLM direct prediction
│   │   ├── hybrid.py          # Predict-verify-repair loop
│   │   └── embedding.py       # Ordinal embedding optimization
│   ├── prompts/               # VLM prompt templates
│   ├── tasks/                 # Evaluation task runners (T1, T2, T3)
│   ├── scripts/               # CLI tools
│   └── tests/                 # Unit tests (70+ tests)
│
└── images/                    # Example rendered images
```

## Key Concepts

### CLEVR Questions
- Questions are generated from templates with functional program representations
- Parameters: Size, Color, Material, Shape, Relation
- Outputs: JSON with question text, answer, and functional program

### ORDINAL-SPATIAL Benchmark

**Tolerance-Based Comparison (tau)**:
- `a <_tau b` iff `a < b × (1 - tau)`
- `a ~=_tau b` iff `|a - b| ≤ tau × max(a, b)`
- `a >_tau b` iff `a > b × (1 + tau)`
- Presets: strict (0.05), standard (0.10), relaxed (0.20)

**Quaternary Relative Relations (QRR)**:
- Compares distances between two pairs: `dist(A,B)` vs `dist(C,D)`
- Metrics: DIST_3D, DIST_2D, DEPTH_GAP, SIZE_RATIO

**Ternary Clock Relations (TRR)**:
- Directional relations using 12-hour clock model
- Target position relative to reference axis

**Tasks**:
- T1-Q: QRR classification
- T1-C: TRR classification (hour + quadrant)
- T2: Constraint extraction from images
- T3: Ordinal reconstruction

## Development Workflows

### Image Generation (Blender)

```bash
# Setup: Add image_generation to Blender's Python path
echo $PWD/image_generation >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth

# Render images (CPU)
cd image_generation
blender --background --python render_images.py -- --num_images 10

# Render with GPU acceleration
blender --background --python render_images.py -- --num_images 10 --use_gpu 1

# Output: output/images/*.png, output/CLEVR_scenes.json
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

# Evaluate predictions
python -m ordinal_spatial.scripts.evaluate \
    --predictions results/predictions.json \
    --ground-truth results/ground_truth.json \
    --task t1-q
```

### Running Tests

```bash
# Run all tests
python -m pytest ordinal_spatial/tests/ -v

# Run specific test file
python -m pytest ordinal_spatial/tests/test_dsl.py -v

# Run with coverage
pytest --cov=ordinal_spatial ordinal_spatial/tests/
```

## Code Conventions

### Style Guidelines
- **Indentation**: 2 spaces (not tabs)
- **Line length**: 80 characters max
- **Naming**: snake_case for functions, PascalCase for classes
- **Type hints**: Full type annotations in ordinal_spatial module

### Docstrings
The ordinal_spatial module uses bilingual docstrings (English + Chinese) with Google/NumPy style:

```python
def compute_qrr(obj_a, obj_b, obj_c, obj_d, metric, tau):
    """
    Compute Quaternary Relative Relation between two object pairs.
    计算两对物体之间的四元相对关系。

    Args:
        obj_a: First object of pair 1
        obj_b: Second object of pair 1
        ...

    Returns:
        Comparator: LT, APPROX, or GT
    """
```

### Design Patterns

**Pydantic Models** for data validation:
```python
class ObjectSpec(BaseModel):
    id: str
    position_3d: List[float]
    shape: str
    color: str
    size: str
```

**Dataclasses** for configuration:
```python
@dataclass
class T1Config:
    tau: float = 0.10
    output_dir: Optional[str] = None
```

**Enums** for type-safe options:
```python
class Comparator(Enum):
    LT = "<"
    APPROX = "~="
    GT = ">"
```

### Import Organization
```python
# Standard library
from typing import Dict, List, Optional
from dataclasses import dataclass

# Third-party
import numpy as np
from pydantic import BaseModel

# Project imports
from ordinal_spatial.dsl.comparators import Comparator, compare
```

## Dependencies

### External Tools
- **Blender 2.78+**: Required for image rendering
- **Python 3.5+**: For original CLEVR code
- **Python 3.7+**: Recommended for ordinal_spatial module

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

### Adding a New Baseline
1. Create new file in `ordinal_spatial/baselines/`
2. Implement the baseline class with `predict()` method
3. Register in `ordinal_spatial/baselines/__init__.py`
4. Add tests in `ordinal_spatial/tests/`

### Adding a New Question Template
1. Create template file in `question_generation/CLEVR_1.0_templates/`
2. Define text templates, program templates, and constraints
3. Update `metadata.json` if adding new functions

### Modifying Scene Generation
1. Edit `image_generation/render_images.py` for rendering changes
2. Update `image_generation/data/` for new shapes/materials
3. Scene ground truth is output to `output/CLEVR_scenes.json`

### Adding a New Evaluation Metric
1. Add metric class in `ordinal_spatial/evaluation/metrics.py`
2. Follow pattern of existing T1/T2/T3 metrics
3. Add tests for the new metric

## Git Workflow

1. Fork the repo and create your branch from `master`
2. Add tests for new code
3. Update documentation for API changes
4. Ensure test suite passes
5. Follow coding style guidelines
6. Complete the Contributor License Agreement

## Key Files Reference

| File | Purpose |
|------|---------|
| `image_generation/render_images.py` | Main Blender rendering script |
| `question_generation/generate_questions.py` | Question synthesis engine |
| `ordinal_spatial/dsl/schema.py` | Core data models (ObjectSpec, OSD) |
| `ordinal_spatial/dsl/comparators.py` | Tolerance-based comparison algebra |
| `ordinal_spatial/dsl/predicates.py` | QRR/TRR computation functions |
| `ordinal_spatial/evaluation/metrics.py` | All evaluation metrics |
| `ordinal_spatial/evaluation/consistency.py` | Graph-based constraint validation |

## Troubleshooting

### Blender Python Path Issues
Ensure the `clevr.pth` file is in the correct site-packages directory for your Blender version.

### GPU Rendering Issues
Verify CUDA is properly installed and use `--use_gpu 1` flag.

### Import Errors in ordinal_spatial
Install all dependencies: `pip install -r ordinal_spatial/requirements.txt`
