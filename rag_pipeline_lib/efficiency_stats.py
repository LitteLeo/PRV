"""
效率画像统计：ParseIndex 失败、JSON 解析失败、fail-safe 触发。
由评测脚本设置当前样本的 stats，core/reranker 通过 get_efficiency_stats() 打点，无需改调用签名。
"""
from contextvars import ContextVar

efficiency_stats_context: ContextVar = ContextVar("efficiency_stats_context", default=None)


def get_efficiency_stats():
    """返回当前上下文中的效率统计 dict（若有）。用于 core/reranker 打点。"""
    return efficiency_stats_context.get()


def create_efficiency_stats():
    """创建并返回一个用于单样本统计的 dict，评测脚本在每样本开始时调用。"""
    return {
        # Fail-safe / 解析鲁棒性
        "parse_index_fail": 0,
        "parse_index_total": 0,
        "json_fail": 0,
        "json_total": 0,
        "failsafe_triggered": False,
        # 闭环稳定性：Bridge evidence retention
        "bridge_retention_checks": 0,
        "bridge_retention_ok": 0,
        # 闭环稳定性：Recovery (PARTIAL/FAILED 后恢复)
        "recovery_req_ids_partial_failed": [],  # req_id 列表，曾出现过 PARTIAL/FAILED
        "recovery_attempts": 0,
        "recovery_successes": 0,
        # 闭环稳定性：False fact / 不可证实事实
        "total_facts": 0,
        "failed_facts": 0,
        "partial_facts": 0,
        # 部署/效率：单样本峰值显存 MB（仅当评测进程使用 GPU 时有效；vLLM 服务端显存需另测）
        "peak_vram_mb": None,
    }
