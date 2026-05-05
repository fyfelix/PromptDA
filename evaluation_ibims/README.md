# PromptDA iBims 官方评估

`evaluation_ibims/` 是 PromptDA 适配后的 iBims 官方评估 pipeline。它只负责：

- 读取已经生成的 synthetic iBims manifest 和 raw depth；
- 使用 `promptda.promptda.PromptDA` 做 RGB-D 推理；
- 保存 iBims 官方脚本要求的 `*_results.mat`；
- 准备官方评估 workspace 并汇总官方 stdout 指标。

本目录不包含、也不导入 `generate_raw_depth.py` 或 `validate_block_mask.py`。

## 输入数据

默认读取：

```text
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_<level>.jsonl
```

manifest 每行需要包含：

```text
sample_id
rgb
raw_depth
depth
depth-range
depth_scale
```

其中 `rgb`、`raw_depth`、`depth` 可以是绝对路径，也可以是相对 manifest 所在目录的路径。

官方评估还需要 iBims 数据集自带文件：

```text
data/ibims1/imagelist.txt
data/ibims1/ibims1_core_mat/*.mat
data/ibims1/evaluation_scripts/evaluate_ibims.py
```

## 数据处理口径

推理数据处理按当前项目 `evaluation/infer.py` 的方式实现：

- RGB 用 OpenCV 读取，BGR 转 RGB，归一化到 `[0, 1]`。
- 当 RGB 最长边大于 `--input-size` 时等比例缩小，目标高宽取 14 的倍数。
- raw depth 用 `cv2.IMREAD_UNCHANGED` 读取，转 `float32` 后除以 manifest 的 `depth_scale`。
- raw depth 中非有限值、负值、超过 `depth-range[1]` 的值置 0。
- PromptDA 输入为 RGB tensor 和 prompt depth tensor；输出 resize 回 GT depth 原始形状。
- 保存到官方 MAT 前，非有限值或 `<=0` 的预测写为 `NaN`。

## 环境

服务器上建议先激活 conda 环境：

```bash
conda activate <your-env>
pip install -r requirements.txt
pip install -r evaluation_ibims/requirements.txt
pip install -e .
```

`run_all.sh` 默认使用当前环境里的 `python`。如需指定：

```bash
PYTHON_BIN=python ./evaluation_ibims/run_all.sh
```

## 运行

在仓库根目录执行：

```bash
./evaluation_ibims/run_all.sh ckpts/promptda_vitl.ckpt vitl
```

常用小样本 smoke：

```bash
IBIMS_ROOT=data/ibims1 \
LEVELS="easy" \
MAX_SAMPLES=1 \
INPUT_SIZE=1008 \
./evaluation_ibims/run_all.sh ckpts/promptda_vitl.ckpt vitl
```

只推理不评估：

```bash
SKIP_EVAL=true ./evaluation_ibims/run_all.sh ckpts/promptda_vitl.ckpt vitl
```

复用已有预测只评估：

```bash
RUN_DIR=evaluation_ibims/output/ibims_promptda_vitl_20260505_120000 \
SKIP_INFER=true \
./evaluation_ibims/run_all.sh ckpts/promptda_vitl.ckpt vitl
```

也可以直接运行单个 manifest：

```bash
python evaluation_ibims/infer_to_mat.py \
  --manifest data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_easy.jsonl \
  --model-path ckpts/promptda_vitl.ckpt \
  --encoder vitl
```

## 输出

默认每次运行写入：

```text
evaluation_ibims/output/ibims_<model>_<timestamp>/
```

目录结构：

```text
run_args.json
predictions/<level>/<sample_id>_results.mat
predictions/<level>/infer_args.json
official_eval/<level>/workspace/
official_eval/<level>/official_eval_stdout.txt
eval_summary.csv
```

`eval_summary.csv` 来自官方脚本 stdout 中的 `Results:` 指标块。
