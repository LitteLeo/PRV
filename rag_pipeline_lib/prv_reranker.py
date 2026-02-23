"""
PRV 重排模块（prv_reranker.py）

在召回（retrieve_context）与事实提取（extract_facts）之间，对 top-k 文档做动态选择与排序。
与 DynamicRAG 的文档选择能力对齐：同一模型输入 query + 文档列表，输出 [1],[2],... 等标识。
"""
import re
import config
from rag_pipeline_lib import llm_adapter
from typing import List, Dict, Any

try:
    from rag_pipeline_lib.efficiency_stats import get_efficiency_stats
except ImportError:
    def get_efficiency_stats():
        return None


def _search_hits_to_retrieved_content(search_hits: List[Dict[str, Any]]) -> str:
    """
    将 REAP 检索结果转为 DynamicRAG 风格的「Retrieved Content」字符串。
    格式：每行为 "i. Topic: title\nContent: text"，与 inference.py get_prompt_docs 一致。
    """
    lines = []
    for i, hit in enumerate(search_hits):
        title = hit.get("title", "")
        text = hit.get("contents", hit.get("text", str(hit)))
        lines.append(f"{i + 1}. Topic: {title}\nContent: {text}")
    return "Retrieved Content:\n" + "\n".join(lines)


def _parse_rerank_response(response_text: str, n_docs: int) -> List[int]:
    """
    从模型输出中解析文档标识，返回 1-based 下标列表（有序）。
    与 DynamicRAG inference.py map_function 一致：re.findall(r'\[(\d+)\]', item)。
    若解析为空或含 "none"，且无有效数字，返回空列表（调用方将 fallback 到原始列表）。
    """
    if not response_text:
        return []
    text_lower = response_text.strip().lower()
    if "none" in text_lower and n_docs > 0:
        # 模型明确说不需要文档时，先尝试解析是否有 [1],[2]；若没有则返回空
        ids = [int(num) for num in re.findall(r"\[(\d+)\]", response_text)]
        if not ids:
            return []
    ids = [int(num) for num in re.findall(r"\[(\d+)\]", response_text)]
    # 只保留合法下标，且去重保序（按出现顺序）
    seen = set()
    result = []
    for i in ids:
        if 1 <= i <= n_docs and i not in seen:
            seen.add(i)
            result.append(i)
    return result


def rerank_documents(query: str, search_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    PRV 重排：对检索到的文档做动态选择与排序，再交给事实提取使用。

    Args:
        query: 当前子任务查询（检索用的 search_query）。
        search_hits: E5 返回的文档列表，每项含 'contents'、'title' 等。

    Returns:
        重排后的文档列表（子集且有序）。若未启用重排、无文档、或解析失败则回退为原列表。
    """
    if not getattr(config, "USE_PRV_RERANK", False):
        return search_hits
    if not search_hits:
        return search_hits

    retrieved_content = _search_hits_to_retrieved_content(search_hits)
    response_text = llm_adapter.rerank_documents(query, retrieved_content)

    ids = _parse_rerank_response(response_text, len(search_hits))
    stats = get_efficiency_stats()
    if stats is not None:
        stats["parse_index_total"] = stats.get("parse_index_total", 0) + 1
        if not ids:
            stats["parse_index_fail"] = stats.get("parse_index_fail", 0) + 1
            stats["failsafe_triggered"] = True
    if not ids:
        # 解析为空或模型输出 None：保留原顺序，避免丢失文档（fail-safe）
        return search_hits
    return [search_hits[i - 1] for i in ids]
