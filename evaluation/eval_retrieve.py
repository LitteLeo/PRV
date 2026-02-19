"""
评测专用：仅在当前题的 ctxs 上做 E5 相似度排序，不查外部检索服务。
知识库 = 当前 question 的 ctxs，对子问题 q_t 在 ctxs 上取 top-k 再送入重排/事实提取。
"""
import os
import sys
from typing import List, Dict, Any, Optional

# 保证可导入 PRV 根模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import config
from search.simple_encoder import SimpleEncoder

_encoder: Optional[SimpleEncoder] = None


def get_e5_encoder() -> SimpleEncoder:
    """懒加载 E5 编码器，用于评测时在 ctxs 上做相似度排序。"""
    global _encoder
    if _encoder is None:
        path = getattr(config, "E5_ENCODER_PATH", None) or os.getenv("E5_ENCODER_PATH")
        if not path:
            raise RuntimeError("E5_ENCODER_PATH not set in config or env for eval retrieval.")
        _encoder = SimpleEncoder(model_name_or_path=path, max_length=512)
        if torch.cuda.is_available():
            _encoder.to("cuda:0")
    return _encoder


def _ctx_to_passage(c: Dict[str, Any]) -> str:
    """将一条 ctx（title + text/contents）拼成一段文本供 E5 编码。"""
    title = c.get("title", "")
    text = c.get("text", c.get("contents", ""))
    if title and text:
        return f"{title} {text}"
    return text or title


def _ctxs_to_search_hits(ctxs: List[Dict], indices: List[int]) -> List[Dict[str, str]]:
    """按 indices 顺序从 ctxs 取出并转为 search_hits 格式（title, contents）。"""
    hits = []
    for i in indices:
        c = ctxs[i]
        title = c.get("title", "")
        text = c.get("text", c.get("contents", ""))
        hits.append({"title": title, "contents": text})
    return hits


def rank_ctxs_by_query(
    query: str,
    ctxs: List[Dict[str, Any]],
    k: int = 40,
    encoder: Optional[SimpleEncoder] = None,
) -> List[Dict[str, str]]:
    """
    在「仅当前题 ctxs」上做 E5 相似度排序，返回按相关性排序的 top-k 条，格式为 search_hits。

    Args:
        query: 子问题或主问题
        ctxs: 当前题的 ctxs，每项为 {title, text} 或 {title, contents}
        k: 返回条数（按相似度截断，避免高相关 doc 被按位置截掉）
        encoder: 可选，不传则用 get_e5_encoder()

    Returns:
        list of {title, contents}，已按 E5 相似度降序，长度 min(k, len(ctxs))
    """
    if not ctxs:
        return []
    encoder = encoder or get_e5_encoder()
    passages = [_ctx_to_passage(c) for c in ctxs]
    # 编码：query 用 query 前缀，passages 用 passage 前缀（在 SimpleEncoder 内）
    q_vec = encoder.encode_queries([query])
    p_vecs = encoder.encode_passages(passages)
    # L2 已归一化，内积即余弦相似度
    q_vec = q_vec.float()
    p_vecs = p_vecs.float()
    if q_vec.dim() == 1:
        q_vec = q_vec.unsqueeze(0)
    scores = (q_vec @ p_vecs.T).squeeze(0)
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)
    k = min(k, len(ctxs))
    _, top_indices = torch.topk(scores, k=k)
    if top_indices.device != torch.device("cpu"):
        top_indices = top_indices.cpu()
    idx_list = top_indices.tolist()
    return _ctxs_to_search_hits(ctxs, idx_list)
