"""
REAP框架检索服务启动脚本（start_e5_server_main.py）

本模块启动HTTP检索服务，提供RESTful API接口供REAP框架调用。
检索服务使用E5Searcher进行向量检索，支持异步处理提高并发性能。

功能：
- 启动HTTP服务器（使用Starlette框架）
- 提供POST接口接收检索请求
- 使用异步处理提高并发性能
- 对应REAP论文公式1：D_t = Retriever(q_t; C)的服务端实现
"""
import os
import asyncio
import traceback
import sys
import random
sys.path.insert(0, 'src')

from typing import List, Dict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from concurrent.futures import ThreadPoolExecutor

from search.e5_searcher import E5Searcher
from logger_config import logger


async def search(request: Request):
    """
    检索API接口处理函数
    
    功能：接收HTTP POST请求，执行向量检索，返回Top-K文档列表。
    这是REAP框架中FE模块的检索服务端实现，对应论文公式1：D_t = Retriever(q_t; C)
    
    请求格式：
    {
        "query": "查询文本",
        "k": 5  # Top-K数量（可选，默认从环境变量TOP_K读取）
    }
    
    响应格式：
    [
        {
            "doc_id": 123,
            "score": 0.95,
            "contents": "文档内容",
            "title": "文档标题"
        },
        ...
    ]
    """
    payload = await request.json()
    query: str = payload['query']
    # Get 'k' from payload, default to TOP_K env var or 5
    top_k_env = int(os.getenv('TOP_K', 5))
    k: int = payload.get('k', top_k_env)

    response_q = asyncio.Queue()
    # Pass query and k to the model_queue
    await request.app.model_queue.put(((query, k), response_q))
    output = await response_q.get()
    return JSONResponse(output)


async def server_loop(q):
    #searcher: E5Searcher = E5Searcher(
    #    index_dir='/home/lfy/data/corpus_embeddings/2wikimqa/',
    #    model_name_or_path='/home/lfy/projects/models/e5-large-v2',
    #    verbose=True # Ensure E5Searcher returns document content
    #)

    searcher: E5Searcher = E5Searcher(
        index_dir='/home/lfy/data/corpus_embeddings/2wikimqa/',  # 改成你的实际索引目录
        model_name_or_path='/home/lfy/projects/models/e5-large-v2',  # 改成你的实际模型路径
        verbose=True
    )

    # Default k from environment, primarily for warmup or if not specified in request
    default_top_k = int(os.getenv('TOP_K', 5))
    logger.info(f'E5Searcher initialized, ready to serve requests, default_top_k={default_top_k}, verbose=True')
    
    # Warmup the server
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        app.executor,
        searcher.batch_search, 
        [f'test query {random.random()} {i}' for i in range(min(64, default_top_k * 2))], 
        default_top_k
    )

    while True:
        try:
            (request_data, response_q) = await q.get()
            query, k_val = request_data

            logger.info(f"Processing query: '{query}' with k={k_val}")

            loop = asyncio.get_event_loop()
            results_list: List[List[Dict]] = await loop.run_in_executor(
                app.executor,
                searcher.batch_search, 
                [query],
                k_val
            )
            
            await response_q.put(results_list[0] if results_list else [])

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            logger.error(traceback.format_exc())
            if 'response_q' in locals() and response_q:
                try:
                    await response_q.put({"error": str(e)})
                except Exception as e_resp:
                    logger.error(f"Error sending error to response_q: {e_resp}")


app = Starlette(
    routes=[
        Route("/", search, methods=["POST"]),
    ],
)


@app.on_event("startup")
async def startup_event():
    q = asyncio.Queue()
    app.model_queue = q
    app.executor = ThreadPoolExecutor(max_workers= 10) 
    asyncio.create_task(server_loop(q))

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app, 'executor') and app.executor:
        app.executor.shutdown(wait=True)
