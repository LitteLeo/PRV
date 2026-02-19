# REAP框架代码结构梳理文档

本文档详细梳理了REAP（Recursive Evaluation and Adaptive Planning）框架的代码结构，按照论文的方法流程组织，帮助理解整个框架的实现。

## 目录

1. [框架概述](#框架概述)
2. [核心模块详解](#核心模块详解)
3. [检索模块详解](#检索模块详解)
4. [LLM适配器模块](#llm适配器模块)
5. [提示词模块](#提示词模块)
6. [完整执行流程](#完整执行流程)
7. [关键公式对应](#关键公式对应)

---

## 框架概述

REAP框架通过"分解-迭代规划-事实提取-合成"的闭环流程，实现多跳问答（MHQA）的精准推理。核心是**子任务规划器（SP）**与**事实提取器（FE）**的递归协同。

### 三大核心模块

1. **Decomposer模块**：初始查询分解
2. **SP模块（子任务规划器）**：包含Plan Updater和Re-Planner两个子模块
3. **FE模块（事实提取器）**：文档检索和事实提取
4. **Synthesizer模块**：答案合成

### 代码结构

```
REAP/
├── rag_pipeline_lib/          # 核心RAG流程库
│   ├── core.py                # 核心功能模块（Decomposer、FE、SP、Synthesizer）
│   ├── pipeline.py            # 主流程控制（迭代循环）
│   ├── prompts.py             # LLM提示词定义
│   ├── llm_adapter.py         # LLM调用适配器
│   └── llm_providers/         # LLM提供商实现
│       ├── vllm_utils.py      # vLLM提供商
│       └── openai_utils.py    # OpenAI提供商
├── search/                    # 检索服务模块
│   ├── search_utils.py        # HTTP检索接口
│   ├── e5_searcher.py         # E5向量检索器
│   ├── simple_encoder.py      # 文本编码器
│   └── start_e5_server_main.py # 检索服务启动脚本
├── config.py                  # 配置文件
└── utils.py                   # 工具函数
```

---

## 核心模块详解

### 1. core.py - 核心功能模块

这是REAP框架的核心实现，包含所有关键功能函数。

#### 1.1 `retrieve_context(query: str) -> list[dict]`

**功能**：文档检索（FE模块的检索组件）

- **对应论文**：公式1 - `D_t = Retriever(q_t; C)`
- **作用**：从语料库C中检索与子任务查询q_t相关的Top-K文档
- **调用位置**：`retrieve_and_extract_facts`函数中

#### 1.2 `analyze_and_decompose_query(query: str) -> dict`

**功能**：阶段1 - 初始查询分解（Decomposer模块）

- **输入**：用户的复杂多跳查询Q
- **输出**：初始任务计划P₀ = {p₁, p₂, ..., pₙ}
  - 每个子任务pᵢ包含：`idᵢ`、`qᵢ`、`depsᵢ`
- **示例**：
  ```python
  # 输入："歌曲《Week Without You》演唱者的生日是什么时候？"
  # 输出：
  {
    "user_goal": "查找歌曲演唱者的生日",
    "requirements": [
      {
        "requirement_id": "req1",
        "question": "《Week Without You》的演唱者是谁？",
        "depends_on": null
      },
      {
        "requirement_id": "req2",
        "question": "req1中标识的演唱者生日是什么时候？",
        "depends_on": "req1"
      }
    ]
  }
  ```

#### 1.3 `retrieve_and_extract_facts(search_query, requirement, collected_facts) -> dict`

**功能**：阶段2.2 - FE处理Actions，提取结构化事实

- **对应论文**：
  - 公式1：`D_t = Retriever(q_t; C)` - 文档检索
  - 公式7：`f_t = M_θ(ExtractF | q_t, D_t, F_{t-1})` - 事实提取
  - 公式8：`f_t = (s_t, e_t, r_t, l_t)` - 结构化事实定义

- **流程**：
  1. 文档检索：调用`retrieve_context`获取相关文档D_t
  2. LLM分析：调用LLM提取结构化事实
  3. 返回事实：包含statement、evidence、reasoning、fulfillment_level

- **输出格式**：
  ```python
  {
    "reasoned_facts": [
      {
        "statement": "《Week Without You》的演唱者是Miley Cyrus",
        "direct_evidence": ["Document 1: 'Miley Ray Hemsworth ... performed the song Week Without You'"],
        "reasoning": "文档1明确提及Miley Cyrus与该歌曲的关联",
        "fulfills_requirement_id": "req1",
        "fulfillment_level": "DIRECT_ANSWER"  # 或 "PARTIAL_CLUE" 或 "FAILED_EXTRACT"
      }
    ]
  }
  ```

#### 1.4 `update_plan(query, collected_facts, pending_requirements) -> dict`

**功能**：阶段2.3 - SP更新计划（Plan Updater子模块）

- **触发条件**：当lₜ=DirectAnswer（理想场景）
- **对应论文**：公式5 - `P_t, Actions_{t+1} = SP(P_{t-1}, F_t, Q)`
- **核心操作**：
  1. **事实替换**：用新事实中的具体实体替换未完成子任务中的抽象占位符
     - 示例：`"What is the birthday of the performer identified in req1?"` 
     → `"What is the birthday of Miley Cyrus?"`
  2. **计划分叉**：若子任务提取到多个答案，复制后续依赖子任务形成并行分支

- **输出格式**：
  ```python
  {
    "decision": {
      "next_step": "CONTINUE_SEARCH",  # 或 "SYNTHESIZE_ANSWER"
      "updated_plan": [...],  # 更新后的任务计划P_t
      "next_actions": [...]   # 下一轮可执行动作Actions_{t+1}
    }
  }
  ```

#### 1.5 `replan_questions(query, collected_facts, pending_requirements) -> dict`

**功能**：阶段2.3 - SP更新计划（Re-Planner子模块）

- **触发条件**：当lₜ=PartialClue/Failed（非理想场景）
- **对应论文**：公式5 - `P_t, Actions_{t+1} = SP(P_{t-1}, F_t, Q)`
- **核心操作**：
  1. **实用充分性评估**：判断部分线索是否足够推进后续推理
  2. **范围化计划修复**：
     - 局部问题：优化查询（如补充实体名称）
     - 系统性问题：修剪无效分支+注入新子任务

- **输出格式**：
  ```python
  {
    "analysis": {
      "problem_diagnosis": "局部问题：子查询表述模糊",
      "recovery_strategy": "优化查询，补充实体名称"
    },
    "decision": {
      "next_step": "CONTINUE_SEARCH",
      "updated_plan": [...],
      "next_actions": [...]
    }
  }
  ```

#### 1.6 `synthesize_final_answer(query, collected_facts) -> str`

**功能**：阶段3 - 答案合成（Synthesizer模块）

- **对应论文**：公式4 - `A = M_θ(Synthesize | Q, F_final)`
- **输入**：原始查询Q和最终事实列表F_final
- **输出**：简洁的最终答案A

---

### 2. pipeline.py - 主流程控制

这是REAP框架的主执行函数，协调所有模块完成完整的推理流程。

#### 2.1 `run_multistep_pipeline(query, verbose, trace_collector) -> str`

**功能**：REAP框架主执行函数

**完整流程**：

```
┌─────────────────────────────────────────────────────────┐
│  阶段1：初始查询分解（Decomposer模块）                    │
│  analyze_and_decompose_query(query)                     │
│  → 生成初始任务计划P₀                                    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段2：核心迭代循环（SP与FE协同）                        │
│  while iteration_count < max_iterations:                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 子步骤2.1：SP分析状态，确定可执行动作Actionsₜ    │  │
│  │ if last_extraction_was_direct_only:              │  │
│  │     update_plan()      # Plan Updater           │  │
│  │ else:                                           │  │
│  │     replan_questions() # Re-Planner            │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 子步骤2.2：FE处理Actionsₜ，提取结构化事实fₜ      │  │
│  │ 并行执行：retrieve_and_extract_facts()           │  │
│  │ → 生成事实列表{f₁, f₂, ..., fₖ}                  │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 子步骤2.3：SP更新计划与事实                        │  │
│  │ F_t = F_{t-1} ∪ {f₁, f₂, ..., fₖ}                │  │
│  │ P_t = SP(P_{t-1}, F_t, Q)                        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  终止条件检查：                                          │
│  - 所有子任务完成 → 进入阶段3                           │
│  - 达到最大迭代次数 → 进入阶段3                         │
│  - 连续失败 → 进入阶段3                                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  阶段3：答案合成（Synthesizer模块）                       │
│  synthesize_final_answer(query, collected_facts)        │
│  → 生成最终答案A                                         │
└─────────────────────────────────────────────────────────┘
```

**关键状态变量**：

- `pending_requirements`：待完成的子任务列表（当前任务计划P_t）
- `collected_facts`：收集的事实列表F_t（初始为空F₀=∅）
- `last_extraction_was_direct_only`：标记上一轮提取是否全部为DirectAnswer
- `iteration_count`：当前迭代次数（成功完成的迭代数）
- `max_iterations`：最大迭代次数（默认5次）

**迭代终止条件**：

1. ✅ **理想终止**：所有子任务完成，事实列表Fₜ已覆盖原始查询所需信息
2. ⚠️ **超时终止**：迭代次数达到预设上限（默认5次）
3. ❌ **失败终止**：连续多轮提取到PartialClue/Failed，且Re-Planner无法生成有效新子任务

**错误处理机制**：

- 状态快照：每次迭代开始前保存状态
- 状态回滚：迭代失败时回滚到迭代开始前的状态
- 日志追踪：支持追踪所有LLM调用和迭代信息

---

## 检索模块详解

检索模块实现了REAP框架的文档检索功能，对应论文公式1：`D_t = Retriever(q_t; C)`

### 3. search/search_utils.py

**功能**：HTTP检索接口（客户端）

- **核心函数**：`search_by_http(query, k, host, port)`
- **作用**：通过HTTP POST请求调用远程检索服务，获取Top-K相关文档
- **对应REAP阶段**：阶段2.2 - FE模块的文档检索步骤
- **调用位置**：`rag_pipeline_lib/core.py` 中的 `retrieve_context()` 函数

**使用示例**：
```python
# 在core.py的retrieve_context函数中调用
search_hits = search_utils.search_by_http(
    query=query,      # 子任务查询q_t
    k=config.TOP_K,  # Top-K数量（论文中设为5）
    host=config.SEARCH_SERVICE_HOST,
    port=config.SEARCH_SERVICE_PORT
)
```

### 4. search/e5_searcher.py

**功能**：E5向量检索器实现（服务端核心）

- **核心类**：`E5Searcher`
- **主要方法**：
  - `__init__()`：初始化检索器，加载预计算向量索引
  - `batch_search(queries, k)`：批量检索Top-K文档
- **技术栈**：
  - 使用e5-large-v2编码器进行文本编码
  - 使用Faiss库进行高效向量相似度搜索
  - 支持GPU加速和分布式检索（多GPU分片）

**工作流程**：
```
1. 加载预计算的文档向量索引（支持分片加载）
2. 使用SimpleEncoder编码查询文本为向量
3. 通过Faiss进行Top-K向量相似度搜索
4. 返回与查询最相关的Top-K文档列表
```

**特性**：
- ✅ GPU加速：使用Faiss GPU索引提高检索速度
- ✅ 分布式检索：将索引分片到多个GPU，支持大规模索引
- ✅ CPU回退：当GPU不可用时自动回退到CPU

### 5. search/simple_encoder.py

**功能**：文本编码器

- **核心类**：`SimpleEncoder`
- **主要方法**：
  - `__init__()`：加载预训练编码模型（如e5-large-v2）
  - `encode_queries(queries)`：将查询文本列表编码为向量矩阵
- **技术细节**：
  - 使用transformers库加载预训练模型
  - 支持批量编码提高效率
  - 支持多种池化方式（cls、avg、last、weightedavg）

**编码流程**：
```
查询文本 → 添加前缀（"query: "） → Tokenization → 
模型编码 → 池化 → L2归一化 → 向量表示
```

### 6. search/start_e5_server_main.py

**功能**：检索服务启动脚本（服务端入口）

- **作用**：启动HTTP服务器，提供RESTful API接口
- **技术栈**：使用Starlette框架实现异步HTTP服务
- **API接口**：
  - **路径**：`POST /`
  - **请求格式**：
    ```json
    {
        "query": "查询文本",
        "k": 5  // 可选，默认从环境变量TOP_K读取
    }
    ```
  - **响应格式**：
    ```json
    [
        {
            "doc_id": 123,
            "score": 0.95,
            "contents": "文档内容",
            "title": "文档标题"
        },
        ...
    ]
    ```

**服务架构**：
```
HTTP请求 → Starlette应用 → 异步队列 → 
线程池执行器 → E5Searcher.batch_search() → 返回结果
```

**特性**：
- ✅ 异步处理：使用asyncio提高并发性能
- ✅ 线程池：使用ThreadPoolExecutor执行CPU密集型检索任务
- ✅ 预热机制：服务启动时执行预热查询，优化首次响应时间

---

## LLM适配器模块

### 7. llm_adapter.py

**功能**：统一的LLM调用接口

**核心函数**：

1. `analyze_query(query)` → 查询分解（Decomposer模块）
2. `extract_facts(query, active_requirement, known_facts, retrieved_documents)` → 事实提取（FE模块）
3. `update_plan(query, collected_facts, pending_requirements)` → 计划更新（Plan Updater）
4. `replan_conditions(query, collected_facts, pending_requirements)` → 计划重新规划（Re-Planner）
5. `generate_final_answer(query, facts)` → 答案合成（Synthesizer模块）

**特性**：

- 支持多种LLM提供商（vLLM、OpenAI等）
- 自动追踪所有LLM调用（通过`@traceable_llm_call`装饰器）
- 统一的错误处理

---

## 提示词模块

### 8. prompts.py

**功能**：定义所有LLM调用的提示词

**提示词分类**：

1. **SYSTEM_PROMPT_QUERY_ANALYSIS**：查询分解提示词（Decomposer模块）
2. **SYSTEM_PROMPT_FACT_EXTRACTION**：事实提取提示词（FE模块）
3. **SYSTEM_PROMPT_PLAN_UPDATER**：计划更新提示词（Plan Updater子模块）
4. **SYSTEM_PROMPT_CONDITION_REPLAN**：计划重新规划提示词（Re-Planner子模块）
5. **SYSTEM_PROMPT_FINAL_ANSWER**：答案合成提示词（Synthesizer模块）

每个提示词都经过精心设计，指导LLM完成特定的任务。

---

## 完整执行流程

### 示例：回答"歌曲《Week Without You》演唱者的生日是什么时候？"

```
1. 阶段1：初始查询分解
   └─> analyze_and_decompose_query()
       → P₀ = {
           req1: "《Week Without You》的演唱者是谁？" (deps=null)
           req2: "req1中标识的演唱者生日是什么时候？" (deps=req1)
         }
       → F₀ = ∅

2. 迭代1：
   ├─> SP分析：Actions₁ = {req1} (req1无依赖，可执行)
   ├─> FE提取：检索文档 → 提取事实
   │   → f₁ = {
   │       statement: "演唱者是Miley Cyrus"
   │       evidence: "Document 1: 'Miley Ray Hemsworth ... performed the song Week Without You'"
   │       reasoning: "文档1明确提及Miley Cyrus与该歌曲的关联"
   │       fulfillment_level: "DIRECT_ANSWER"
   │     }
   └─> SP更新：
       ├─> F₁ = F₀ ∪ {f₁}
       ├─> 事实替换：req2的查询更新为"What is the birthday of Miley Cyrus?"
       └─> P₁ = {req2}, Actions₂ = {req2}

3. 迭代2：
   ├─> SP分析：Actions₂ = {req2} (req2依赖已满足)
   ├─> FE提取：检索文档 → 提取事实
   │   → f₂ = {
   │       statement: "Miley Cyrus的生日是1992年11月23日"
   │       evidence: "Document 2: 'Miley Ray Hemsworth was born on November 23, 1992'"
   │       reasoning: "文档2明确提及Miley Cyrus的出生日期"
   │       fulfillment_level: "DIRECT_ANSWER"
   │     }
   └─> SP更新：
       ├─> F₂ = F₁ ∪ {f₂}
       └─> 所有子任务完成 → next_step = "SYNTHESIZE_ANSWER"

4. 阶段3：答案合成
   └─> synthesize_final_answer()
       → A = "1992年11月23日"
```

---

## 关键公式对应

| 公式 | 对应函数 | 说明 |
|------|---------|------|
| **公式1**：`D_t = Retriever(q_t; C)` | `retrieve_context()` | 文档检索 |
| **公式4**：`A = M_θ(Synthesize \| Q, F_final)` | `synthesize_final_answer()` | 答案合成 |
| **公式5**：`P_t, Actions_{t+1} = SP(P_{t-1}, F_t, Q)` | `update_plan()` / `replan_questions()` | 计划更新 |
| **公式7**：`f_t = M_θ(ExtractF \| q_t, D_t, F_{t-1})` | `retrieve_and_extract_facts()` | 事实提取 |
| **公式8**：`f_t = (s_t, e_t, r_t, l_t)` | `retrieve_and_extract_facts()` 返回格式 | 结构化事实定义 |

---

## 多任务微调说明

REAP框架支持多任务微调优化SP模块性能：

- **数据准备**：整合Decomposer、Plan Updater、Re-Planner三个模块的训练数据
- **模型设计**：使用单一规划模型M_φ同时学习三个模块的任务
- **损失函数**：加权联合损失，公式9：
  ```
  L_multi(φ) = Σ_{task∈T} λ_task * E_{(x,y)~D_task}[L_task(M_φ(x), y)]
  ```
- **训练框架**：使用LlamaFactory，采用LoRA技术微调

---

## 总结

REAP框架通过精心设计的模块化架构，实现了：

1. **全局规划能力**：SP模块通过维护完整任务计划与动态调度，避免传统多轮RAG的"局部推理僵局"
2. **事实可靠性**：FE的结构化事实设计（含证据与推理过程）降低幻觉，提升可追溯性
3. **效率与适应性**：轻量级模型替换（Plan Updater用1B参数模型）与迭代次数控制，平衡性能与效率
4. **泛化能力**：多任务微调与潜在线索挖掘，使模型在域外数据集上仍保持优异性能

所有代码都添加了详细的中文注释，按照REAP论文的方法流程组织，便于理解和维护。

