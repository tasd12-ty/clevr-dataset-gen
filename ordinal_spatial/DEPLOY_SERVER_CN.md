# 服务器部署说明（含本地模型 OpenAI 接口）

## 1. 现在支持的部署形态
- `ordinal_spatial` 目录内已内置 `image_generation`（用于重新生成+渲染）。
- `generate_and_render.py` 与 `build_benchmark.py` 会优先使用：
  1. `--img-gen-dir`
  2. `IMG_GEN_DIR`
  3. `ordinal_spatial/image_generation`（默认）
  4. 仓库根目录 `image_generation`（兼容旧布局）

因此把 `ordinal_spatial` 整个目录迁移到服务器即可运行重生成与渲染（前提：服务器有 Blender 5.0+）。

## 2. 本地部署模型（vLLM/OpenAI 兼容）
- 约束抽取脚本支持 `--api-base --model --api-key`。
- 只要你的多模态服务兼容 OpenAI Chat Completions，即可直接替换：
  - `--api-base http://<host>:<port>/v1`
  - `--model <你的模型名>`
  - `--api-key <任意占位或真实 key>`

## 3. 示例命令
```bash
# 生成+渲染（使用内置 image_generation）
PYTHONPATH=.. python3 -m ordinal_spatial.scripts.generate_and_render \
  --output-dir ./results \
  --blender-path /path/to/blender \
  --min-objects 3 --max-objects 4 --scenes-per-count 2 --samples 16

# 单视角 VLM 抽取（OpenAI 兼容本地服务）
PYTHONPATH=.. python3 -m ordinal_spatial.scripts.extract_vlm_single \
  --benchmark-dir ./data/benchmark \
  --split test_iid \
  --model your-mm-model \
  --api-base http://127.0.0.1:8000/v1 \
  --api-key dummy \
  --max-scenes 10
```

## 4. 依赖
- Python >= 3.9
- `pip install -e ..` 或按仓库根目录 `pyproject.toml` 安装依赖
- 关键包：`openai`, `numpy`, `pydantic`, `Pillow`, `scipy`
