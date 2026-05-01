# PromptDA HAMMER 评估

这个目录是为 PromptDA 适配后的 HAMMER 评估 pipeline 副本。原始 pipeline
仍保留在软链接 `run_bs_eval_pipeline/` 下；当前仓库实际可运行的评估脚本放在
这里。

## 适配模型

固定目标模型是 `promptda.promptda.PromptDA`，默认使用项目 README 推荐的
`vitl` encoder。PromptDA 是 RGB-D 模型：HAMMER 的 `rgb` 字段作为图像输入，
指定 raw depth 字段作为 `prompt_depth`。

模型输出是以 meter 为单位的 metric depth。`evaluation/infer.py` 会把
PromptDA 输出 resize 回 GT depth 分辨率，并将每个样本保存为
`HxW float32 .npy`，供 `evaluation/eval.py` 直接读取。

推理阶段默认会对输入 PromptDA 的 raw/prompt depth 做空洞补全：`0`、非有限值、
负值以及 HAMMER `depth-range` 外的像素会用最近有效深度填充。该补洞只作用于
PromptDA 输入，不修改 GT depth，也不改变 `evaluation/eval.py` 里的 valid mask
和指标计算口径。

当前不做 disparity 或 inverse-depth 转换。默认也不需要 alignment，因为
PromptDA 会根据 raw prompt depth 的范围反归一化预测结果。如果要评估非 metric
checkpoint，请在评估阶段显式增加 alignment，并单独记录该次运行设置。

## 数据

HAMMER 默认放在项目路径：

```text
data/HAMMER/test.jsonl
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
`evaluation/output/2026-05-01_12-30-00/`。

常用环境变量覆盖：

```bash
DATASET_PATH=data/HAMMER/test.jsonl \
OUTPUT_DIR=evaluation/output \
MAX_SAMPLES=0 \
FILL_PROMPT_DEPTH_HOLES=true \
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
Prompt depth 可视化展示的是实际送入 PromptDA 的深度；默认启用补洞时，即为补洞后
的 prompt depth。设置 `FILL_PROMPT_DEPTH_HOLES=false` 可关闭补洞，复现原始
`0` 空洞输入行为。
设置 `MAX_SAMPLES=N` 可只运行并评估数据集前 N 条样本；默认 `MAX_SAMPLES=0`
表示评估全部样本。
只有在明确希望把预测值 clamp 到 HAMMER `depth-range` 时，才设置
`CLAMP_PREDICTION=true`。

## 说明

- `evaluation/eval.py`、`evaluation/dataset.py` 和
  `evaluation/utils/metric.py` 复制自原始 pipeline，并保留固定 HAMMER 指标。
- PromptDA 推理按单样本执行。建议保持 `BATCH_SIZE=1`，除非只是想让 DataLoader
  对路径做 batch。
- RGB 使用 OpenCV 读取，并在推理前从 BGR 转为 RGB。
- raw depth PNG 会除以 `depth_scale=1000`，因此保存的预测结果保持 meter 单位。
