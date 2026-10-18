# SGMEA Framework

这是当前 SGMEA 主线实验的可复现代码快照，包含两部分：

1. **Candidate generation**：Name、Attribute、Image、Relation 四个视图的相似度，融合得到初始 Top-20；随后用 Top-20 的跨图对应关系更新 relation/graph support，并迭代排序，最后把前 10 个候选交给 agent。
2. **Agent loop**：独立的 `planner` 和 `planner_choice`，按需调用 `attribute`、`relation`、`visual` 工具；每轮更新当前判断和已获得的 receipt，允许 planner 直接 `finish`，或最多再获取两种模态后结束。

代码中没有提交数据集、模型权重、图片特征、运行结果、缓存或密钥。它们需要在运行机器上通过环境变量提供。

## 目录

```text
candidate/                 四视图候选与关系更新代码
agent/                     planner/choice、三类 modality agent、运行器
scripts/prepare_agent_bridge.py
configs/candidate_protocol.json
configs/legacy_requirements.txt
```

## 环境

建议 Python 3.10/3.11、CUDA 与本机 PyTorch 匹配。先安装 PyTorch，再安装 Python 依赖：

```bash
python -m venv .venv
source .venv/bin/activate
# 按 CUDA 版本安装 torch/torchvision
pip install -r requirements.txt
```

模型和数据默认使用相对路径，也可以用 `.env.example` 中的变量改成绝对路径：

- `SGMEA_MMKGB_ROOT`：MMKG/DBP15K 原始数据根目录；默认 `data/mmkg`。
- `SGMEA_DBP_ROOT`：仅旧式 DBP relation 控制脚本使用；默认 `data/DBP15K`。
- `SGMEA_EMBEDDING_MODEL`：Qwen3-Embedding-8B 路径。
- `SGMEA_TEXT_MODEL`：LaBSE 路径。
- `SGMEA_VL_MODEL`：agent 使用的 Qwen3-VL 模型路径。
- `SGMEA_AGENT_DEPENDENCY_ROOT`：agent 运行时依赖/processor 路径。
- `SGMEA_PROJECT_ROOT`、`SGMEA_BASE_ROOT`、`SGMEA_V69_ROOT`：工程和运行输出目录。

## 数据布局

`SGMEA_MMKGB_ROOT` 下需要保留原始发布的目录结构：

```text
data/mmkg/
  DBP15K/{zh_en,ja_en,fr_en}/
  FBDB15K/norm/
  FBYG15K/norm/
  pkls/                         # 图像特征 pickle
```

DBP15K 的名称翻译文件和五个数据集的实体/属性/关系文件由候选脚本按原始布局读取。不要把这些数据或图像特征提交到 GitHub。

## 生成四视图候选

主入口是 `candidate/e2e_raw_single_loop.py`。`generate` 阶段不读取对齐标签，默认按四视图生成、用 residual relation update 迭代两步；当前实验协议可显式写出为：

```bash
export SGMEA_PROJECT_ROOT=$PWD
export SGMEA_MMKGB_ROOT=/path/to/mmkg
export SGMEA_EMBEDDING_MODEL=/path/to/Qwen3-Embedding-8B
export SGMEA_TEXT_MODEL=/path/to/LaBSE

python candidate/e2e_raw_single_loop.py \
  --root outputs/candidate-main \
  --stage generate \
  --datasets zh_en ja_en fr_en FBDB15K FBYG15K \
  --update residual \
  --steps 4 \
  --readout compatibility \
  --scope ordinary \
  --attribute-mode typed \
  --calendar-kernel overlap \
  --fusion geometric \
  --correspondence balanced \
  --support-statistic degree_cosine \
  --support-norm rowmax \
  --support-k 20
```

每个数据集完成后会在 `outputs/candidate-main/done/` 写入 checkpoint，排名在 `outputs/candidate-main/rankings/`，协议和代码 hash 在输出根目录。重新运行同一命令可以续跑；改变协议时请使用新的 `--root`。

可选的 `verify` 阶段用于复用已经审计过的 feature root；正式从原始数据生成时应使用上面的 `generate`。`evaluate` 只在单独提供 labels root 后运行，不参与候选生成：

```bash
python candidate/e2e_raw_single_loop.py \
  --root outputs/candidate-main --stage evaluate \
  --datasets zh_en ja_en fr_en FBDB15K FBYG15K \
  --labels-root /path/to/labels
```

## 接入 agent

先把候选输出桥接成 agent 输入。桥接脚本只读取候选排名和原始数据，不读取对齐标签：

```bash
export SGMEA_RANK_ROOT=$PWD/outputs/candidate-main/rankings/all
export SGMEA_BRIDGE_ROOT=$PWD/outputs/agent_bridge
python scripts/prepare_agent_bridge.py
```

然后准备 agent 运行根目录。它必须包含：

```text
$SGMEA_V69_ROOT/
  tasks.json
  inputs/<cohort>/<task>.json
  image_indices/<cohort>.json
  previews/<cohort>/<task>.json
  priors/<cohort>/<task>.json
  prompts/{planner,planner_choice,attribute_scope_grounded,
           relation_scope_grounded,visual_scope_grounded}.system.txt
  runtime_model.json
  code_snapshot/              # 与输入一起生成的代码快照和 hash
  input_sha256.json
  code_sha256.json
```

这些输入包含桥接后的前 10 候选、原始名称/属性/关系和图像索引；本仓库不上传具体实验输入。设置模型后启动 8-worker runner：

```bash
export SGMEA_PROJECT_ROOT=$PWD
export SGMEA_V69_ROOT=$PWD/outputs/agent-run
export SGMEA_VL_MODEL=/path/to/Qwen3-VL-8B-Instruct
export SGMEA_AGENT_DEPENDENCY_ROOT=/path/to/agent-dependencies
python agent/run_agent.py launch
```

`launch` 会先校验输入与代码 hash，然后启动 8 个 worker。默认 GPU 调度沿用当前实验的成对映射 `(0,1),(0,1),(2,3),(2,3),(4,5),(4,5),(6,7),(6,7)`；可以通过编辑运行器或设置可见 GPU 环境适配机器。完成后统计：

```bash
python agent/run_agent.py evaluate
```

运行器默认保留 `no_acquisition` 和 `no_acquisition_rethink` 对照；如只跑主方法，可设置 `SGMEA_V69_INCLUDE_BASELINE=0`。单数据集/小规模试跑可设置 `SGMEA_V69_TASK_LIMIT=10`。

## 检查

不需要加载模型即可做静态检查：

```bash
python scripts/check_repo.py
```

它会编译 Python 文件、检查仓库中是否残留 `/gly` 绝对路径或疑似密钥，并列出被 Git 忽略的大文件类型。
