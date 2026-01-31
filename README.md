# CLEVR Dataset Generation

This is the code used to generate the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/) as described in the paper:

**[CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](http://cs.stanford.edu/people/jcjohns/clevr/)**
 <br>
 <a href='http://cs.stanford.edu/people/jcjohns/'>Justin Johnson</a>,
 <a href='http://home.bharathh.info/'>Bharath Hariharan</a>,
 <a href='https://lvdmaaten.github.io/'>Laurens van der Maaten</a>,
 <a href='http://vision.stanford.edu/feifeili/'>Fei-Fei Li</a>,
 <a href='http://larryzitnick.org/'>Larry Zitnick</a>,
 <a href='http://www.rossgirshick.info/'>Ross Girshick</a>
 <br>
 Presented at [CVPR 2017](http://cvpr2017.thecvf.com/)

Code and pretrained models for the baselines used in the paper [can be found here](https://github.com/facebookresearch/clevr-iep).

You can use this code to render synthetic images and compositional questions for those images, like this:

<div align="center">
  <img src="images/example1080.png" width="800px">
</div>

**Q:** How many small spheres are there? <br>
**A:** 2

**Q:**  What number of cubes are small things or red metal objects? <br>
**A:**  2

**Q:** Does the metal sphere have the same color as the metal cylinder? <br>
**A:** Yes

**Q:** Are there more small cylinders than metal things? <br>
**A:** No

**Q:**  There is a cylinder that is on the right side of the large yellow object behind the blue ball; is there a shiny cube in front of it? <br>
**A:**  Yes

If you find this code useful in your research then please cite

```
@inproceedings{johnson2017clevr,
  title={CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning},
  author={Johnson, Justin and Hariharan, Bharath and van der Maaten, Laurens
          and Fei-Fei, Li and Zitnick, C Lawrence and Girshick, Ross},
  booktitle={CVPR},
  year={2017}
}
```

## Blender Version Compatibility

| Blender Version | Status | Notes |
|-----------------|--------|-------|
| 2.78c | Fully Supported | Original development version |
| 2.79b | Fully Supported | Last version with original API |
| 2.80 - 3.x | Supported | Requires v5 resource files |
| 4.x | Supported | Requires v5 resource files |
| **5.0+** | **Supported** | Requires v5 resource files |

Code has been updated to automatically detect Blender version and use the appropriate API.

## Step 1: Generating Images

First we render synthetic images using [Blender](https://www.blender.org/), outputting both rendered images as well as a JSON file containing ground-truth scene information for each image.

### Setup

Blender ships with its own installation of Python which is used to execute scripts that interact with Blender. You'll need to add the `image_generation` directory to the Python path of Blender's bundled Python.

#### For Blender 2.78-2.79 (Legacy)
```bash
echo $PWD/image_generation >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth
```

#### For Blender 5.0+ (Modern)
```bash
# Linux
echo $PWD/image_generation >> ~/.config/blender/5.0/python/lib/python3.11/site-packages/clevr.pth

# macOS
echo $PWD/image_generation >> /Applications/Blender.app/Contents/Resources/5.0/python/lib/python3.11/site-packages/clevr.pth

# Windows (run from Blender Python)
# See image_generation/README_BLENDER5.md for details
```

### Blender 5.0+ Quick Start

For Blender 5.0+, first generate the compatible resource files:

```bash
cd image_generation

# Option 1: One-click setup (Linux/macOS)
./setup_blender5.sh /path/to/blender

# Option 2: Manual setup
blender --background --python create_base_scene_blender5.py
blender --background --python create_materials_blender5.py
blender --background --python create_shapes_blender5.py
```

Then render images:

```bash
blender --background --python render_images.py -- \
    --base_scene_blendfile data/base_scene_v5.blend \
    --material_dir data/materials_v5 \
    --shape_dir data/shapes_v5 \
    --num_images 10
```

### Blender 2.78-2.79 (Legacy)

```bash
cd image_generation
blender --background --python render_images.py -- --num_images 10
```

### GPU Acceleration

If you have a GPU with CUDA/OptiX/HIP installed, you can accelerate rendering:

```bash
blender --background --python render_images.py -- --num_images 10 --use_gpu 1
```

Supported GPU backends (Blender 5.0+):
- NVIDIA CUDA
- NVIDIA OptiX
- AMD HIP
- Intel OneAPI

After rendering, you should have images stored in `output/images` like these:

<div align="center">
  <img src="images/img1.png" width="260px">
  <img src="images/img2.png" width="260px">
  <img src="images/img3.png" width="260px">
  <br>
  <img src="images/img4.png" width="260px">
  <img src="images/img5.png" width="260px">
  <img src="images/img6.png" width="260px">
</div>

The file `output/CLEVR_scenes.json` will contain ground-truth scene information for all newly rendered images.

You can find [more details about image rendering here](image_generation/README.md).

## Step 2: Generating Questions

Next we generate questions, functional programs, and answers for the rendered images generated in the previous step.
This step takes as input the single JSON file containing all ground-truth scene information, and outputs a JSON file
containing questions, answers, and functional programs for the questions in a single JSON file.

You can generate questions like this:

```bash
cd question_generation
python generate_questions.py
```

The file `output/CLEVR_questions.json` will then contain questions for the generated images.

You can [find more details about question generation here](question_generation/README.md).

## Additional Modules

### ORDINAL-SPATIAL Benchmark

This repository also includes the **ORDINAL-SPATIAL** module for evaluating Vision-Language Models on ordinal spatial reasoning tasks. See [ordinal_spatial/README.md](ordinal_spatial/README.md) for details.

```bash
# Quick start
pip install -r ordinal_spatial/requirements.txt
python -m ordinal_spatial.scripts.generate_dataset --small --output-dir ./data
```

## Project Structure

```
clevr-dataset-gen/
├── image_generation/              # Blender-based image rendering
│   ├── render_images.py           # Main rendering script (version-aware)
│   ├── utils.py                   # Blender utilities (version-aware)
│   ├── collect_scenes.py          # Scene collection utility
│   ├── create_base_scene_blender5.py   # Generate v5 base scene
│   ├── create_materials_blender5.py    # Generate v5 materials
│   ├── create_shapes_blender5.py       # Generate v5 shapes
│   ├── setup_blender5.sh          # One-click Blender 5.0 setup
│   ├── README.md                  # Image generation guide
│   ├── README_BLENDER5.md         # Blender 5.0 compatibility guide
│   └── data/
│       ├── base_scene.blend       # Scene for Blender 2.78
│       ├── base_scene_v5.blend    # Scene for Blender 5.0+
│       ├── properties.json        # Object property definitions
│       ├── CoGenT_A.json          # CoGenT condition A
│       ├── CoGenT_B.json          # CoGenT condition B
│       ├── materials/             # Materials for Blender 2.78
│       ├── materials_v5/          # Materials for Blender 5.0+
│       ├── shapes/                # Shapes for Blender 2.78
│       └── shapes_v5/             # Shapes for Blender 5.0+
│
├── question_generation/           # Question synthesis
│   ├── generate_questions.py      # Main question generation script
│   ├── question_engine.py         # Template processing engine
│   ├── metadata.json              # Functional programming language specs
│   ├── synonyms.json              # Natural language synonyms
│   ├── README.md                  # Question generation guide
│   └── CLEVR_1.0_templates/       # Question templates (~9 files)
│       ├── zero_hop.json          # Direct attribute queries
│       ├── one_hop.json           # Single relation queries
│       ├── two_hop.json           # Two relation queries
│       ├── three_hop.json         # Three relation queries
│       ├── comparison.json        # Attribute comparison
│       ├── compare_integer.json   # Count comparison
│       ├── same_relate.json       # Same-attribute queries
│       ├── single_and.json        # AND logic queries
│       └── single_or.json         # OR logic queries
│
├── ordinal_spatial/               # Spatial reasoning benchmark module
│   ├── dsl/                       # Domain-Specific Language
│   │   ├── schema.py              # Core data models
│   │   ├── predicates.py          # Spatial predicates
│   │   └── comparators.py         # Comparison functions
│   ├── evaluation/                # Metrics and consistency checking
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── consistency.py         # Consistency checker
│   ├── generation/                # Constraint and dataset generation
│   │   ├── constraint_extractor.py
│   │   ├── degeneracy_checker.py
│   │   └── difficulty_control.py
│   ├── baselines/                 # Baseline implementations
│   │   ├── oracle.py              # Oracle baseline
│   │   ├── vlm_direct.py          # Direct VLM baseline
│   │   ├── embedding.py           # Embedding baseline
│   │   └── hybrid.py              # Hybrid baseline
│   ├── prompts/                   # VLM prompt templates
│   ├── tasks/                     # Evaluation task runners (T1/T2/T3)
│   ├── scripts/                   # CLI tools
│   │   ├── generate_dataset.py    # Dataset generation
│   │   ├── run_baseline.py        # Run baselines
│   │   ├── evaluate.py            # Evaluation script
│   │   └── visualize.py           # Visualization
│   ├── tests/                     # Unit tests
│   └── README.md                  # Module documentation
│
├── images/                        # Example rendered images
├── CLAUDE.md                      # AI assistant guide
├── CLEVR数据集生成工具完整报告.md   # Complete Chinese documentation
├── LICENSE                        # BSD License
└── README.md                      # This file
```

## Documentation

- [Image Generation Guide](image_generation/README.md)
- [Blender 5.0 Compatibility Guide](image_generation/README_BLENDER5.md)
- [Question Generation Guide](question_generation/README.md)
- [ORDINAL-SPATIAL Benchmark](ordinal_spatial/README.md)
- [Complete Chinese Documentation](CLEVR数据集生成工具完整报告.md)

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.
