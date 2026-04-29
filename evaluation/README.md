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

在仓库根目录执行：

```bash
./evaluation/run_eval.sh /path/to/model.ckpt
```

完整位置参数：

```bash
./evaluation/run_eval.sh <checkpoint_or_hf_model_id> [encoder=vitl] [raw_type=d435] [cleanup_npy=true]
```

常用环境变量覆盖：

```bash
DATASET_PATH=data/HAMMER/test.jsonl \
OUTPUT_DIR=outputs/hammer_promptda_vitl_d435 \
PYTHON_BIN=python3 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
./evaluation/run_eval.sh /path/to/model.ckpt vitl d435 false
```

`checkpoint_or_hf_model_id` 可以是本地 `.ckpt` 路径，也可以是 Hugging Face
模型 id，例如 `depth-anything/prompt-depth-anything-vitl`。使用 Hugging Face id
时可能会在运行时下载权重。

## 输出

输出目录包含：

```text
args.json
eval_args.json
*.npy
all_metrics_<timestamp>_False.csv
mean_metrics_<timestamp>_False.json
```

设置 `SAVE_VIS=true` 可额外保存 RGB、prompt depth、prediction 的可视化 JPG。
只有在明确希望把预测值 clamp 到 HAMMER `depth-range` 时，才设置
`CLAMP_PREDICTION=true`。

## 说明

- `evaluation/eval.py`、`evaluation/dataset.py` 和
  `evaluation/utils/metric.py` 复制自原始 pipeline，并保留固定 HAMMER 指标。
- PromptDA 推理按单样本执行。建议保持 `BATCH_SIZE=1`，除非只是想让 DataLoader
  对路径做 batch。
- RGB 使用 OpenCV 读取，并在推理前从 BGR 转为 RGB。
- raw depth PNG 会除以 `depth_scale=1000`，因此保存的预测结果保持 meter 单位。
