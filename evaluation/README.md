# PromptDA HAMMER / ClearPose 评估

这个目录是为 PromptDA 适配后的 HAMMER / ClearPose 评估 pipeline。原始 pipeline
仍保留在软链接 `run_bs_eval_pipeline/` 下；当前仓库实际可运行的评估脚本放在这里。

## 适配模型

固定目标模型是 `promptda.promptda.PromptDA`，默认使用项目 README 推荐的
`vitl` encoder。PromptDA 是 RGB-D 模型：数据集的 RGB 图像作为图像输入，
指定 raw depth 作为 `prompt_depth`。

模型输出是以 meter 为单位的 metric depth。`evaluation/infer.py` 会把
PromptDA 输出 resize 回 GT depth 分辨率，并将每个样本保存为
`HxW float32 .npy`，供 `evaluation/eval.py` 直接读取。

默认数据集仍指向 HAMMER 已经离线补洞后的 D435 raw depth JSONL。通过
`DATASET_PATH` 可以切换到 ClearPose JSONL。推理阶段不再额外执行 prompt depth
空洞补全；GT depth、`evaluation/eval.py` 里的 valid mask 和指标计算口径保持不变。

当前不做 disparity 或 inverse-depth 转换。默认也不需要 alignment，因为
PromptDA 会根据 raw prompt depth 的范围反归一化预测结果。如果要评估非 metric
checkpoint，请在评估阶段显式增加 alignment，并单独记录该次运行设置。

## 数据集

HAMMER 默认使用已补洞的 D435 JSONL：

```text
data/HAMMER/test_filled_d435.jsonl
```

数据集 JSONL 由复制过来的 `HAMMERDataset` 读取，字段需要包含：

```text
rgb
d435_depth
l515_depth
tof_depth
depth
depth-range
```

用 `raw_type` 选择 raw depth 来源：`d435`、`l515` 或 `tof`。

ClearPose 可通过 `DATASET_PATH` 指向 JSONL，例如：

```text
data/clearpose/test.jsonl
```

ClearPose JSONL 字段需要包含：

```text
rgb
rgb-suffix
raw_depth-suffix
depth-suffix
depth-range
```

ClearPose 只支持 `raw_type=d435`。推理保存和评估读取的 `.npy` 名称共用同一规则：

```text
HAMMER: scene#frame-stem.npy
ClearPose: set#scene#frame-stem.npy
```

例如：

```text
data/clearpose/set2/scene4/000709-color.png
-> set2#scene4#000709-color.npy
```

## 运行

先安装仓库根依赖，再安装评估脚本的补充依赖：

```bash
pip install -r requirements.txt
pip install -r evaluation/requirements.txt
```

在仓库根目录执行：

```bash
./evaluation/run_eval.sh
```

完整位置参数：

```bash
./evaluation/run_eval.sh [checkpoint_or_hf_model_id=ckpts/promptda_vitl.ckpt] [encoder=vitl] [raw_type=d435] [cleanup_npy=false]
```

默认 checkpoint 路径是 `ckpts/promptda_vitl.ckpt`。`OUTPUT_DIR` 是输出根目录；
每次运行会在其下新建一个时间戳目录，例如
`evaluation/output/hammer_2026-05-01_12-30-00/` 或
`evaluation/output/clearpose_2026-05-01_12-30-00/`。

常用环境变量覆盖：

```bash
DATASET_PATH=data/HAMMER/test_filled_d435.jsonl \
OUTPUT_DIR=evaluation/output \
MAX_SAMPLES=0 \
PYTHON_BIN=python3 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
./evaluation/run_eval.sh /path/to/model.ckpt vitl d435 false
```

ClearPose 示例：

```bash
DATASET_PATH=data/clearpose/test.jsonl \
OUTPUT_DIR=evaluation/output \
MAX_SAMPLES=0 \
PYTHON_BIN=python3 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
./evaluation/run_eval.sh /path/to/model.ckpt vitl d435 false
```

`checkpoint_or_hf_model_id` 可以是本地 `.ckpt` 路径，也可以是 Hugging Face
模型 id，例如 `depth-anything/prompt-depth-anything-vitl`。使用 Hugging Face id
时可能会在运行时下载权重。

## 输出

每次运行的时间戳目录包含：

```text
args.json
eval_args.json
predictions/*.npy
visualizations/*_promptda_vis.jpg
all_metrics_<timestamp>_False.csv
mean_metrics_<timestamp>_False.json
```

默认会保存 RGB、prompt depth、prediction 的可视化 JPG；设置 `SAVE_VIS=false`
可关闭可视化输出。
Prompt depth 可视化展示的是实际送入 PromptDA 的深度；默认情况下该深度来自
`data/HAMMER/test_filled_d435.jsonl` 指向的已补洞 raw depth 文件。
设置 `MAX_SAMPLES=N` 可只运行并评估数据集前 N 条样本；默认 `MAX_SAMPLES=0`
表示评估全部样本。
只有在明确希望把预测值 clamp 到数据集 `depth-range` 时，才设置
`CLAMP_PREDICTION=true`。

## 说明

- `evaluation/eval.py`、`evaluation/dataset.py` 和
  `evaluation/utils/metric.py` 复制自原始 pipeline，并保留原有指标计算口径。
- PromptDA 推理按单样本执行。建议保持 `BATCH_SIZE=1`，除非只是想让 DataLoader
  对路径做 batch。
- RGB 使用 OpenCV 读取，并在推理前从 BGR 转为 RGB。
- raw depth PNG 会除以 `depth_scale=1000`，因此保存的预测结果保持 meter 单位。
