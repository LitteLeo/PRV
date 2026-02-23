# PRV：规划–重排–校验闭环推理框架

本目录为 **PRV（Planning–Reranking–Verification）** 的实现，基于 REAP 代码复制并增加「召回后重排」集成与统一模型配置。

## 设计说明

- 框架设计文档（推荐先读）：**[PRV框架设计说明.md](../PRV框架设计说明.md)**（仓库根目录）
- 流程：规划（REAP 分解与迭代）→ **重排**（对 E5 召回 top-k 做模型动态选择/排序）→ 校验（事实提取与答案合成）

## 相对 REAP 的改动

1. **统一模型**：所有 LLM 任务（含重排）使用 **REAP-all-lora**（`config.VLLM_LLM_MODEL`，默认 `/home/lfy/projects/models/REAP-all-lora`）。
2. **重排集成**：
   - 在 `rag_pipeline_lib/core.py` 的 `retrieve_and_extract_facts` 中，在 `retrieve_context()` 之后、格式化文档并调用事实提取之前，若 `config.USE_PRV_RERANK` 为真，则调用 `prv_reranker.rerank_documents(search_query, search_hits)` 对文档做动态选择与排序。
   - 重排与 **DynamicRAG** 的文档选择能力对齐：同一模型输入 query + 文档列表，输出 `[1],[2],...` 等标识，解析后得到有序子集再送入事实提取。
3. **新增模块**：
   - `rag_pipeline_lib/prv_reranker.py`：构建重排 prompt、调用 LLM、解析输出并返回重排后的文档列表。
   - `rag_pipeline_lib/prompts.py` 中新增 `SYSTEM_PROMPT_RERANK_DOCS`、`USER_PROMPT_RERANK_DOCS`。
   - `rag_pipeline_lib/llm_providers/vllm_utils.py` 中新增 `vllm_rerank_documents`；`llm_adapter.py` 中新增 `rerank_documents`（可追踪）。
4. **配置**：
   - `config.py`：`USE_PRV_RERANK`（默认 True）、所有 VLLM 模型路径统一为 REAP-all-lora，并修正原 REAP 中 `VLLM_EXTRACT_FACTS_MODEL` 等未正确赋值的问题。

## 运行方式

与 REAP 相同：先启动 E5 检索服务与 vLLM（加载 REAP-all-lora），再运行 pipeline。环境变量可覆盖：

- `VLLM_LLM_MODEL`：模型路径（默认 REAP-all-lora）
- `USE_PRV_RERANK`：`true` / `false`，是否启用重排
- `MAX_DOCUMENT_CHARS`：送入事实提取的文档总字符数上限（默认 24000，约 6k tokens，预留其余给 prompt/query）。设为 `0` 表示不截断。见下「超出模型上下文」说明。

其他配置见 `config.py` 及根目录 `PRV框架设计说明.md`。

## 超出模型上下文与当前处理方式

当 Top-N 较大（如 200/300/500）时，拼接后的文档可能超过模型上下文（如 LLaMA3-8B 的 8192 tokens）。

- **当前实现（顺序截断）**：在重排之后、事实提取之前，对文档列表按 **字符数上限** 做截断：只保留前若干条，使格式化后的总长度 ≤ `MAX_DOCUMENT_CHARS`（默认 24000 字符，约 6k tokens），避免单次调用 LLM 时超出上下文。逻辑在 `core._truncate_hits_to_fit_context`，在 `retrieve_and_extract_facts` 与 `extract_facts_given_hits` 以及评测单轮路径中统一使用。
- **未实现**：真正的 **滑动窗口**（将文档分成多段，每段分别做事实提取再合并）尚未实现；若需严格复刻「Sun et al. 滑动窗口」，需在 FE 前对文档做分块、多次调用 extract_facts 并合并 `reasoned_facts`。
- **重排阶段**：重排器当前仍接收完整 Top-N 文档列表；若 N 极大导致重排 prompt 超长，需在重排前对文档做截断或分块（可复用 `MAX_DOCUMENT_CHARS` 或单独配置）。

## 与 DynamicRAG 同数据集的分数验证（不走 E5）

评测集为「问题 + ctxs」（如 100 条预检索文档），与 DynamicRAG 的 `eval_data` 一致。为公平对比，**本评测不使用 E5**：直接用评测集自带的 `ctxs`，经 PRV 重排 → 事实提取 → 答案合成，输出与 DynamicRAG 同格式的 JSON，再用 DynamicRAG 的 `evaluate.py` 算 EM/F1。

- **脚本**：`scripts/run_prv_v2_batch.sh`
- **流程**：读入与 `run_8b_v2_batch.sh` 相同的 JSONL → 对每条用 `evaluation/run_prv_eval.py`（ctxs → 重排 → 提取事实 → 合成）→ 写 `results/prv_8b_v2_${dataset}.json` → 调用 `DynamicRAG/evaluate.py` 出分。
- **环境变量**：`DATA_DIR`（评测数据目录，默认 `/home/lfy/data/eval_data`）、`DYNAMICRAG_DIR`（DynamicRAG 路径，用于 evaluate.py，默认 `PRV/../DynamicRAG`）、`OUT_DIR`、`LOG_DIR`、`EXP_DIR` 可覆盖。
- **运行**：在 PRV 目录下执行 `bash scripts/run_prv_v2_batch.sh`（需已启动 vLLM 加载 REAP-all-lora；本评测不需要 E5 服务）。
