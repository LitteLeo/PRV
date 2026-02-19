"""
REAP框架检索工具模块（search_utils.py）

本模块提供HTTP接口调用检索服务的功能，用于REAP框架的文档检索步骤。
检索服务通常运行在独立服务器上，使用e5-large-v2等编码器进行向量检索。

功能：
- 通过HTTP POST请求调用远程检索服务
- 返回Top-K相关文档列表
- 对应REAP论文公式1：D_t = Retriever(q_t; C)
"""
import requests
from typing import List, Dict
from logger_config import logger


def search_by_http(query: str, k: int, host: str = 'localhost', port: int = 8090) -> List[Dict]:
    """
    通过HTTP接口调用检索服务，获取与查询相关的Top-K文档
    
    功能：这是REAP框架中FE模块的文档检索组件，对应论文公式1：
    D_t = Retriever(q_t; C)
    
    检索服务通常使用e5-large-v2编码器进行向量检索，返回与查询最相关的Top-K文档。
    论文中设置每次检索Top-5文档（由参数k控制），确保信息覆盖度。
    
    Args:
        query: 检索查询字符串，通常是子任务的查询q_t
        k: 返回的Top-K文档数量（论文中设为5）
        host: 检索服务的主机地址
        port: 检索服务的端口号
        
    Returns:
        List[Dict]: 检索到的文档列表，每个文档包含：
            - 'doc_id' (int): 文档ID
            - 'score' (float): 相似度分数
            - 'contents' (str): 文档内容（如果verbose模式开启）
            - 'title' (str): 文档标题（如果有）
        如果检索失败，返回空列表[]
    """
    url = f"http://{host}:{port}/"
    payload = {'query': query, 'k': k}
    try:
        # 发送HTTP POST请求到检索服务
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Error Connecting: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout Error: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"An error occurred: {req_err}")
    except Exception as e:
        logger.error(f"Failed to get a response. Status code: {response.status_code if 'response' in locals() else 'N/A'}, Error: {e}")
    return []
