# PromptDA HAMMER / ClearPose / DREDS / TRansPose 评估

这个目录是 PromptDA 的轻量评估导出目录。推理脚本固定加载
`promptda.promptda.PromptDA`，使用 RGB 图像和指定 raw depth 作为
`prompt_depth`，输出 meter 单位的 metric depth。

```text
evaluation/
├── dataset.py
├── infer.py
├── eval.py
├── run_hammer.sh
├── run_clearpose.sh
├── run_dreds.sh
├── run_bs_transpose.sh
├── requirements.txt
└── utils/
```

链路边界：

1. `infer.py` 读取 JSONL 和 checkpoint，写出 `predictions/*.npy`。
2. `eval.py` 读取 `predictions/*.npy`，计算指标并写出 CSV/JSON。
3. `run_*.sh` wrapper 负责选择数据集、组织输出目录、顺序执行推理和评估。

## 数据集格式

HAMMER JSONL 每行需要包含：

```text
rgb
d435_depth
l515_depth
tof_depth
depth
depth-range
```

`camera_type` 选择 `d435`、`l515` 或 `tof` 作为 PromptDA prompt depth。
HAMMER 的 raw/GT depth 按 16-bit PNG 深度处理，`depth_scale=1000.0`。

ClearPose JSONL 每行需要包含：

```text
rgb
rgb-suffix
raw_depth-suffix
depth-suffix
depth-range
```

ClearPose 按序列展开，固定 `raw-type=d435`，`depth_scale=1000.0`。

DREDS JSONL 使用与 CDM 导出目录一致的序列字段：

```text
rgb
rgb-suffix
raw_depth-suffix
depth-suffix
depth-range
```

DREDS 按序列展开，每个 sequence 最多取 50 帧，EXR 深度单位已经是 meter，
`depth_scale=1.0`。脚本会设置 `OPENCV_IO_ENABLE_OPENEXR=1` 以启用 OpenCV EXR 读取。

TRansPose JSONL 每行需要包含：

```text
rgb
l515_depth
depth
seq_name  # optional
depth-range  # optional
```

TRansPose 固定 `raw-type=l515`。`l515_depth` 和 `depth` 都按毫米 PNG 读取，
`depth_scale=1000.0`，默认有效深度范围为 `0.1-6.0m`。如果 JSONL 提供
`depth-range`，会覆盖默认范围。推理和评估优先用 `seq_name` 查找同一个
`predictions/<seq_name>.npy`；缺少 `seq_name` 时才从 RGB 路径生成兜底名称。

样本命名约定：

```text
HAMMER:    scene#frame-stem.npy
ClearPose: dir1#dir2#frame-stem.npy
DREDS:     dir1#dir2#frame-stem.npy
TRansPose: <seq_name>.npy；无 seq_name 时为 ancestor_frame-stem.npy
```

## 安装

先安装仓库根依赖，再安装评估脚本补充依赖：

```bash
pip install -r requirements.txt
pip install -r evaluation/requirements.txt
```

当前系统 `python3` 如果缺少 `cv2`，请通过 `PYTHON_BIN=/path/to/python` 指向已安装项目依赖的环境。

## HAMMER

```bash
DATASET_PATH=data/HAMMER/test_filled_d435.jsonl \
OUTPUT_DIR=/tmp/promptda_hammer_eval \
MAX_SAMPLES=0 \
bash evaluation/run_hammer.sh /path/to/model.ckpt vitl d435 false
```

参数：

```text
bash evaluation/run_hammer.sh <checkpoint_or_hf_model_id> [encoder=vitl] [camera_type=d435] [cleanup_npy=false]
```

默认输出目录：

```text
<checkpoint_dir>/hammer_<checkpoint_stub>_data_<camera_type>/
```

## ClearPose

ClearPose 固定使用 `raw-type=d435`：

```bash
DATASET_PATH=data/clearpose/test.jsonl \
OUTPUT_DIR=/tmp/promptda_clearpose_eval \
bash evaluation/run_clearpose.sh /path/to/model.ckpt vitl false
```

参数：

```text
bash evaluation/run_clearpose.sh <checkpoint_or_hf_model_id> [encoder=vitl] [cleanup_npy=false]
```

默认输出目录：

```text
<checkpoint_dir>/clearpose_<checkpoint_stub>_data_d435/
```

## DREDS

DREDS 支持 `catknown`、`catnovel`、`all`：

```bash
DREDS_KNOWN_JSONL=data/DREDS/test_std_catknown.jsonl \
DREDS_NOVEL_JSONL=data/DREDS/test_std_catnovel.jsonl \
OUTPUT_ROOT=/tmp/promptda_dreds_eval \
bash evaluation/run_dreds.sh /path/to/model.ckpt vitl all false
```

参数：

```text
bash evaluation/run_dreds.sh <checkpoint_or_hf_model_id> [encoder=vitl] [variant=all] [cleanup_npy=false]
```

说明：

- `variant=catknown` 使用 `DREDS_KNOWN_JSONL`。
- `variant=catnovel` 使用 `DREDS_NOVEL_JSONL`。
- `variant=all` 会顺序运行两个 variant，此时使用 `OUTPUT_ROOT`，不能设置 `OUTPUT_DIR`。
- DREDS 的 prediction shape 如与 GT 不一致，`eval.py` 会用 nearest resize 对齐；HAMMER / ClearPose 遇到 shape mismatch 会报错。

默认输出目录：

```text
<checkpoint_dir>/dreds_catknown_<checkpoint_stub>/
<checkpoint_dir>/dreds_catnovel_<checkpoint_stub>/
```

## TRansPose

TRansPose 固定使用 `raw-type=l515`：

```bash
DATASET_PATH=data/TRansPose/sequences/dc_testset.jsonl \
INTRINSICS_PATH=data/TRansPose/sequences/intrinsics.txt \
OUTPUT_DIR=/tmp/promptda_transpose_eval \
bash evaluation/run_bs_transpose.sh /path/to/model.ckpt vitl l515 true
```

参数：

```text
bash evaluation/run_bs_transpose.sh <checkpoint_or_hf_model_id> [encoder=vitl] [camera_type=l515] [cleanup_npy=true]
```

说明：

- `camera_type` 只接受 `l515`。
- 默认 JSONL 为 `data/TRansPose/sequences/dc_testset.jsonl`。
- 默认 intrinsics 为 `data/TRansPose/sequences/intrinsics.txt`。
- `SAVE_VIS=false` 时不要求 intrinsics 文件存在。
- `SAVE_VIS=true` 时会生成 RGB、raw depth、prediction、GT depth、prediction point cloud、GT point cloud 的 3x2 网格可视化，并要求 intrinsics 文件有效。
- TRansPose 保留 PromptDA 自己的模型加载、checkpoint 解析、prompt depth 预处理和 metric depth 输出链路，不使用 CDM 的 `RGBDDepth` 或 `model_registry.yaml`。

默认输出目录：

```text
<checkpoint_dir>/transpose_<dataset_stub>_<checkpoint_stub>_data_l515/
```

## 常用环境变量

```text
DATASET_PATH          HAMMER / ClearPose / TRansPose JSONL 路径
DREDS_KNOWN_JSONL     DREDS catknown JSONL 路径
DREDS_NOVEL_JSONL     DREDS catnovel JSONL 路径
OUTPUT_DIR            单次运行输出目录
OUTPUT_ROOT           DREDS all 模式的输出根目录
INPUT_SIZE            PromptDA max RGB side length，默认 1008
BATCH_SIZE            DataLoader path batch size；TRansPose wrapper 默认 16，其他 wrapper 默认 1
NUM_WORKERS           DataLoader worker 数；TRansPose wrapper 默认 4，其他 wrapper 默认 0
MAX_SAMPLES           最多运行样本数，0 表示全部
SAVE_VIS              true 时保存可视化图；TRansPose 默认 false，其他 wrapper 默认 true
INTRINSICS_PATH       TRansPose 点云可视化 intrinsics 文件路径
CLAMP_PREDICTION      true 时把 prediction clamp 到 dataset depth-range
PYTHON_BIN            Python 可执行文件，默认 python3
```

## 输出

每个输出目录包含：

```text
args.json
eval_args.json
predictions/*.npy
visualizations/*_promptda_vis.jpg
visualizations/*_grid_vis.jpg
all_metrics_<timestamp>_False.csv
mean_metrics_<timestamp>_False.json
```

如果 `cleanup_npy=true`，评估结束后会删除 `predictions/*.npy`，指标 CSV/JSON 保留。
`eval.py` 默认从 `predictions/` 读取，也兼容旧格式中直接放在输出根目录下的 `.npy`。

## 关键约定

- 不使用 CDM 的 `RGBDDepth`、`is_disp` 或 `resize_method` 参数。
- 不改 PromptDA 模型结构、前向逻辑和预处理链路。
- `infer.py` 会把 PromptDA 输出 resize 回 GT depth 分辨率后保存。
- TRansPose 推理和评估都通过 `seq_name` 查找同一个 `<seq_name>.npy`，避免按 RGB 路径推导名称导致不一致。
- `eval.py` 保留现有指标：`L1`、`rmse_linear`、`abs_relative_difference`、`delta4_acc_105`、`delta5_acc110`、`delta1_acc`。
