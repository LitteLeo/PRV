"""
REAP框架文本编码器模块（simple_encoder.py）

本模块实现了基于预训练模型的文本编码功能，用于REAP框架的向量检索。
使用e5-large-v2等编码器将文本转换为向量表示。

功能：
- 使用预训练模型（如e5-large-v2）编码文本
- 支持查询和文档的编码
- 支持批量编码提高效率
"""
import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import List, Dict, Optional, Union
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from logger_config import logger
#from utils import pool, move_to_device, get_detailed_instruct, get_task_def_by_task_name, create_batch_dict
from utils import pool, move_to_device, create_batch_dict


class SimpleEncoder(nn.Module):
    """
    简单文本编码器类
    
    功能：使用预训练模型（如e5-large-v2）将文本编码为向量表示。
    这是REAP框架中FE模块的编码组件，用于将查询和文档转换为向量。
    
    主要功能：
    1. 加载预训练编码模型（如e5-large-v2）
    2. 将文本编码为固定维度的向量
    3. 支持批量编码提高效率
    """
    def __init__(self, model_name_or_path: str, prefix_type: Optional[str] = 'instruction', pool_type: Optional[str] = 'last',
                 task_name: Optional[str] = None, max_length: int = 512):
        """
        初始化编码器
        
        Args:
            model_name_or_path: 预训练模型路径（如e5-large-v2）
            prefix_type: 前缀类型（'instruction'或'query_or_passage'）
            pool_type: 池化类型（'cls'、'avg'、'last'、'weightedavg'）
            task_name: 任务名称（用于生成指令前缀）
            max_length: 最大序列长度
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        base_name: str = model_name_or_path.split('/')[-1]
        self.pool_type = 'avg'
        self.prefix_type = 'query_or_passage'
        self.max_length = max_length
        assert self.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
        assert self.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'

        self.encoder = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            local_files_only=True,  # 添加这一行，强制使用本地文件
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=True,  # 添加这一行，强制使用本地文件
        )

        self.prompt: Optional[str] = None
        if self.prefix_type == 'instruction' and task_name is not None:
            task_def: str = get_task_def_by_task_name(task_name=task_name)
            self.prompt = get_detailed_instruct(task_def)
            logger.info('Set prompt: {}'.format(self.prompt))

        self.encoder.eval()
        logger.info(f'pool_type={self.pool_type}, prefix_type={self.prefix_type}, prompt={self.prompt}')

    def encode_queries(self, queries: List[str], **kwargs) -> torch.Tensor:
        """
        编码查询文本为向量
        
        功能：将查询文本列表编码为向量表示，用于向量检索。
        这是REAP框架中FE模块的编码步骤，对应论文公式1的查询编码部分。
        
        Args:
            queries: 查询文本列表（通常是子任务的查询q_t）
            **kwargs: 其他可选参数（如batch_size）
            
        Returns:
            torch.Tensor: 查询向量矩阵，形状为 [num_queries, embedding_dim]
        """
        if self.prefix_type == 'query_or_passage':
            input_texts = [f'query: {q}' for q in queries]
        else:
            input_texts = [self.prompt + q for q in queries]

        return self._do_encode(input_texts, **kwargs)

    def encode_passages(self, passages: List[str], **kwargs) -> torch.Tensor:
        """
        编码文档/段落为向量（E5 使用 passage: 前缀，与 encode_queries 的 query: 配对）。
        用于评测时在 ctxs 上做相似度排序（知识库=当前题 ctxs）。
        """
        if self.prefix_type == 'query_or_passage':
            input_texts = [f'passage: {p}' for p in passages]
        else:
            input_texts = [self.prompt + p for p in passages]
        return self._do_encode(input_texts, **kwargs)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str], **kwargs) -> torch.Tensor:
        encoded_embeds = []
        batch_size = kwargs.get('batch_size', 128)
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10, disable=len(input_texts) < 128):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts, max_length=self.max_length)
            batch_dict = move_to_device(batch_dict, device=self.encoder.device)

            with torch.amp.autocast('cuda'):
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().to(torch.float16))

        return torch.cat(encoded_embeds, dim=0)

    def to(self, device):
        self.encoder.to(device)
        return self
