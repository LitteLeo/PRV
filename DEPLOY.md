# PRV 部署到服务器与快速测试

## 1. 同步代码到服务器

在**本机**执行，把本地 PRV 推到服务器 `219.216.64.231` 的 `/home/lfy/projects/PRV`：

```bash
# 本机终端（在 PRVRag 或 PRV 的上一级目录执行）
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.idea' \
  /Users/lan/Documents/RAG/PRVRag/PRV/ \
  lfy@219.216.64.231:/home/lfy/projects/PRV/
```

若本机是 PRV 目录：

```bash
cd /Users/lan/Documents/RAG/PRVRag
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.idea' \
  PRV/ \
  lfy@219.216.64.231:/home/lfy/projects/PRV/
```

## 2. 服务器环境与路径

- **项目路径**：`/home/lfy/projects/` 下已有 `DynamicRAG`、`REAP` 等，PRV 放在 `/home/lfy/projects/PRV`。
- **模型路径**：`/home/lfy/projects/models/REAP-all-lora`（PRV 默认已用该路径，无需改）。
- **评测数据**：与 DynamicRAG 一致，例如 `/home/lfy/data/eval_data/`（`nq.jsonl`、`2wikimqa.jsonl` 等）。

### 必须先启动 vLLM（否则会报 Connection error）

PRV 评测**必须**先在一台机子上把 REAP-all-lora 用 vLLM 拉起来，再跑 `run_prv_quick_test.sh` 或 `run_prv_v2_batch.sh`。否则会全部报 `Error calling VLLM ... Connection error`，结果里答案为空、EM=0。

**先开一个终端启动服务（保持运行）：**
```bash
# 在服务器上执行，占满一张卡时可用 --port 8000 只起一个实例
vllm serve /home/lfy/projects/models/REAP-all-lora --host 0.0.0.0 --port 8000
```

**再开一个终端跑评测：**
```bash
cd /home/lfy/projects/PRV
bash scripts/run_prv_quick_test.sh
```

若 vLLM 不在本机或端口不同，可设环境变量后再跑脚本，例如：
```bash
export VLLM_HOST=127.0.0.1   # 或其它机器 IP
export VLLM_PORT=8000
bash scripts/run_prv_quick_test.sh
```

## 3. 先跑几条测一下（推荐）

在服务器上进入 PRV 目录，用少量样本跑一条数据集，确认能跑通再跑全量：

```bash
ssh lfy@219.216.64.231
cd /home/lfy/projects/PRV

# 默认：nq.jsonl 前 5 条，结果在 results/prv_quick_test_nq.json
bash scripts/run_prv_quick_test.sh
```

可改样本数或数据集（环境变量）：

```bash
SAMPLE_SIZE=10 DATASET=2wikimqa.jsonl bash scripts/run_prv_quick_test.sh
```

- 输入：`DATA_DIR` 下的 `nq.jsonl`（默认 `/home/lfy/data/eval_data`，可覆盖）。
- 输出：`results/prv_quick_test_nq.json`、日志在 `logs/prv_quick_test_nq.log`。
- 若同目录下有 `DynamicRAG`，脚本会顺带用 `evaluate.py` 算一下 EM。

## 4. 全量跑

确认快速测试无误后再跑全量：

```bash
cd /home/lfy/projects/PRV
bash scripts/run_prv_v2_batch.sh
```

- 会遍历 nq、2wikimqa、hotpotqa 等，写 `results/prv_8b_v2_${base}.json`，并调用 DynamicRAG 的 `evaluate.py` 出分。
- 环境变量：`DATA_DIR`、`DYNAMICRAG_DIR`（默认 `PRV/../DynamicRAG`）、`OUT_DIR`、`LOG_DIR`、`EXP_DIR` 可按需覆盖。
