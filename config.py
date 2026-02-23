"""
Configuration settings for the RAG pipeline.
Loads sensitive keys from .env file.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- AI Provider Selection (for LLM Generation) ---
# Supported: 'ollama', 'gemini', 'grok', 'deepseek', 'vllm', 'openai'
# AI_PROVIDER = os.getenv('AI_PROVIDER', 'vllm').lower()
AI_PROVIDER = 'vllm'

# --- API Keys (Loaded from .env file or environment variables) ---

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # For OpenAI


# --- VLLM Settings ---
# VLLM_HOST = os.getenv('VLLM_HOST', "http://172.18.144.20")
VLLM_HOST = os.getenv('VLLM_HOST', "http://127.0.0.1")


# --- PRV 统一模型：需与 vLLM 实际加载的模型路径一致，否则会 404 ---
# 常用：REAP-all-merged（LoRA 已合并）、或 REAP-all-lora
VLLM_LLM_MODEL = os.getenv("VLLM_LLM_MODEL", "/home/lfy/projects/models/REAP-all-merged")
VLLM_LLM_PORT = int(os.getenv("VLLM_LLM_PORT", 8000))

# --- VLLM Dedicated Models Configuration ---
# PRV 单模型：所有任务使用同一 REAP-all-lora，设为 False。
VLLM_USE_DEDICATED_MODELS = False

# 各任务模型配置（单模型时均使用默认 VLLM_LLM_MODEL / VLLM_LLM_PORT）
VLLM_ANALYZE_QUERY_MODEL = VLLM_LLM_MODEL
VLLM_ANALYZE_QUERY_PORT = VLLM_LLM_PORT

VLLM_EXTRACT_FACTS_MODEL = VLLM_LLM_MODEL
VLLM_EXTRACT_FACTS_PORT = VLLM_LLM_PORT

VLLM_UPDATE_PLAN_MODEL = VLLM_LLM_MODEL
VLLM_UPDATE_PLAN_PORT = VLLM_LLM_PORT

VLLM_REPLAN_CONDITIONS_MODEL = VLLM_LLM_MODEL
VLLM_REPLAN_CONDITIONS_PORT = VLLM_LLM_PORT

VLLM_GENERATE_FINAL_ANSWER_MODEL = VLLM_LLM_MODEL
VLLM_GENERATE_FINAL_ANSWER_PORT = VLLM_LLM_PORT


# Deprecated, use VLLM_HOST and specific ports instead
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', f"{VLLM_HOST}:{VLLM_LLM_PORT}/v1")

OPENAI_LLM_MODEL = "gpt-4"


# --- LLM for evaluation ---
LLM_MODEL_EVAL = "gpt-4"

# --- Search Service Configuration ---
# SEARCH_SERVICE_HOST = os.getenv('SEARCH_SERVICE_HOST', '172.18.144.21')
# SEARCH_SERVICE_PORT = int(os.getenv('SEARCH_SERVICE_PORT', 8090))
SEARCH_SERVICE_HOST = '127.0.0.1'
SEARCH_SERVICE_PORT = 8090

# --- Proxy Configuration (Optional) ---
# Set PROXY_ENABLED to 'True' or 'true' in .env or environment to enable
PROXY_ENABLED_str = os.getenv('PROXY_ENABLED', 'False').lower()
PROXY_ENABLED = PROXY_ENABLED_str == 'true'
PROXY_URL = os.getenv('PROXY_URL', None) 


# --- RAG Parameters ---
TOP_K = int(os.getenv('TOP_K', 5)) # Number of documents to retrieve

# --- Control Flags ---
SHOW_RETRIEVED_CONTEXT = True  # Set to False to hide context printout

# PRV 重排：召回后是否对 top-k 做模型动态选择与排序（再送入事实提取）
USE_PRV_RERANK = os.getenv("USE_PRV_RERANK", "true").lower() == "true"

# 消融：是否要求 FE 输出结构化证据与满足度标签（false 时仅产出自由文本结论，下游默认 DIRECT_ANSWER）
USE_VERIFICATION_CONSTRAINTS = os.getenv("USE_VERIFICATION_CONSTRAINTS", "true").lower() == "true"

# 评测时「仅从 ctxs 检索」使用的 E5 编码器路径（进程内排序，不依赖 E5 检索服务）
E5_ENCODER_PATH = os.getenv("E5_ENCODER_PATH", "/home/lfy/projects/models/e5-large-v2")

# 送入 LLM 的文档总字符数上限（避免超出上下文，如 LLaMA3-8B 约 8192 tokens，按 ~4 字符/token 预留约 24000 给文档，其余给 prompt/query）
# 设为 0 表示不截断。滑动窗口（多段分别提取再合并）未实现，当前仅做「按顺序截断」。
MAX_DOCUMENT_CHARS = int(os.getenv("MAX_DOCUMENT_CHARS", "24000"))

# --- LLM Model Name (Set based on provider for generation) ---
LLM_MODEL_NAME = None # Initialize
if  AI_PROVIDER == 'vllm':
    LLM_MODEL_NAME = VLLM_LLM_MODEL
elif AI_PROVIDER == 'openai':
    LLM_MODEL_NAME = OPENAI_LLM_MODEL
else:
    raise ValueError(f"Unsupported AI_PROVIDER for LLM: {AI_PROVIDER}.")


