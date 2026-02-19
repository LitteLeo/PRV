"""
REAP框架E5检索器模块（e5_searcher.py）

本模块实现了基于e5-large-v2编码器的向量检索功能，用于REAP框架的文档检索步骤。
使用Faiss库进行高效的向量相似度搜索，支持GPU加速和分布式检索。

功能：
- 加载预计算的文档向量索引
- 使用e5-large-v2编码器编码查询
- 通过Faiss进行Top-K向量检索
- 对应REAP论文公式1：D_t = Retriever(q_t; C)
"""
import glob
import torch
import pandas as pd
import pickle
import numpy as np
import faiss

from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset, load_from_disk

from search.simple_encoder import SimpleEncoder
from logger_config import logger

def load_corpus() -> List[Dict]:
    #pickle_path = '/home/lfy/data/corpus/2wikimqa_corpus.pkl'
    #logger.info(f"Attempting to load pre-serialized corpus from {pickle_path}...")
    pickle_path = '/home/lfy/data/corpus/2wikimqa_corpus.pkl'
    local_path = '/home/lfy/data/corpus/'
    try:
        with open(pickle_path, 'rb') as f:
            corpus_list: List[Dict] = pickle.load(f)
        logger.info(f'Successfully loaded {len(corpus_list)} items from pickled corpus.')
    except FileNotFoundError:
        logger.warning(f"Pickled corpus not found at {pickle_path}. Falling back to loading from disk dataset.")
        local_path = '/home/lfy/data/corpus/'
        try:
            dataset_on_disk: Dataset = load_from_disk(local_path)
            logger.info(f'Loaded {len(dataset_on_disk)} passages from local dataset at {local_path}. Converting to in-memory list...')
            try:
                df = dataset_on_disk.to_pandas()
                corpus_list = df.to_dict('records')
            except Exception as e_pandas:
                logger.error(f"Pandas conversion failed during fallback: {e_pandas}")
                corpus_list = [item for item in dataset_on_disk]
            logger.info(f'Converted dataset to an in-memory list with {len(corpus_list)} items.')
        except FileNotFoundError:
            logger.error(f"Disk dataset not found at {local_path} during fallback. Cannot load corpus.")
            return []
        except Exception as e_load_disk:
            logger.error(f"Error loading dataset from disk during fallback: {e_load_disk}. Cannot load corpus.")
            return []
    except Exception as e:
        logger.error(f"General error loading pickled corpus: {e}. Aborting corpus load.")
        return []
    return corpus_list


def _get_all_shards_path(index_dir: str) -> List[str]:
    path_list = glob.glob('{}/*-shard-*.pt'.format(index_dir))
    if not path_list:
        logger.error(f"No embedding shards found in {index_dir}")
        raise FileNotFoundError(f"No embedding shards found in {index_dir}. Please check the path and file naming convention ('prefix-shard-NUMBER.pt').")

    def _parse_shard_idx(p: str) -> int:
        try:
            return int(p.split('-shard-')[1].split('.')[0])
        except (IndexError, ValueError) as e:
            logger.error(f"Could not parse shard index from filename: {p}. Expected format like 'name-shard-0.pt'. Error: {e}")
            raise ValueError(f"Invalid shard filename format: {p}. Ensure it contains '-shard-INDEX.pt'.") from e


    path_list = sorted(path_list, key=lambda path: _parse_shard_idx(path))
    logger.info('Embeddings path list: {}'.format(path_list))
    return path_list


class E5Searcher:
    """
    E5向量检索器类
    
    功能：使用e5-large-v2编码器进行向量检索，支持GPU加速和分布式检索。
    这是REAP框架中FE模块的检索组件实现，对应论文公式1：D_t = Retriever(q_t; C)
    
    主要功能：
    1. 加载预计算的文档向量索引（支持分片加载）
    2. 使用e5-large-v2编码器编码查询文本
    3. 通过Faiss进行Top-K向量相似度搜索
    4. 返回与查询最相关的Top-K文档
    
    特性：
    - 支持GPU加速（使用Faiss GPU索引）
    - 支持分布式检索（将索引分片到多个GPU）
    - 支持CPU回退（当GPU不可用时）
    """

    def __init__(
            self, index_dir: str,
            model_name_or_path: str = '/path/to/e5-large-v2',
            verbose: bool = False,
    ):
        """
        初始化E5检索器
        
        Args:
            index_dir: 预计算向量索引的目录路径
            model_name_or_path: e5-large-v2模型路径
            verbose: 是否加载完整文档内容到内存（用于详细输出）
        """
        self.model_name_or_path = model_name_or_path
        self.index_dir = index_dir
        self.verbose = verbose

        # --- MODIFICATION FOR GPU SELECTION START ---
        physical_gpu_count: int = torch.cuda.device_count()
        
        #_explicitly_desired_gpu_ids: List[int] = [4, 5, 6, 7] 
        _explicitly_desired_gpu_ids: List[int] = [0]  # 改成 [0] 使用第一个GPU（单卡环境）

        self.gpu_ids_to_use: List[int] = []
        if physical_gpu_count == 0:
            logger.warning("No physical GPUs available. Faiss will run on CPU if faiss-cpu is installed.")
        else:
            for gid in _explicitly_desired_gpu_ids:
                if 0 <= gid < physical_gpu_count:
                    self.gpu_ids_to_use.append(gid)
                else:
                    logger.warning(f"Desired GPU ID {gid} is not available (total physical GPUs: {physical_gpu_count}). It will be skipped.")
            
            if not self.gpu_ids_to_use:
                logger.warning(f"None of the specifically desired GPUs ({_explicitly_desired_gpu_ids}) are available or valid. "
                               "Faiss will attempt to run on CPU if faiss-cpu is installed (and no specific GPUs were usable).")
            else:
                logger.info(f"Script configured to use the following GPUs for Faiss: {self.gpu_ids_to_use}.")
        
        num_selected_gpus: int = len(self.gpu_ids_to_use)
        # --- MODIFICATION FOR GPU SELECTION END ---

        self.encoder: SimpleEncoder = SimpleEncoder(
            model_name_or_path=self.model_name_or_path,
            max_length=64,
        )
        
        # --- MODIFICATION FOR ENCODER PLACEMENT ---
        if self.gpu_ids_to_use:
            encoder_gpu_id = self.gpu_ids_to_use[-1] 
            try:
                self.encoder.to(f'cuda:{encoder_gpu_id}')
                logger.info(f"Encoder model placed on cuda:{encoder_gpu_id}")
            except Exception as e:
                logger.error(f"Failed to move encoder to cuda:{encoder_gpu_id}. Error: {e}. Encoder will remain on its default device.")
        else:
            logger.info("No GPUs selected for use. Encoder model running on CPU (or its default device).")
        # --- MODIFICATION FOR ENCODER PLACEMENT END ---

        try:
            shard_paths = _get_all_shards_path(self.index_dir)
        except FileNotFoundError as e:
            logger.error(f"Initialization failed: {e}")
            raise
            
        all_embeddings_cpu_f32: torch.Tensor = torch.cat(
            [torch.load(p, weights_only=True, map_location='cpu') for p in shard_paths], dim=0
        ).float()
        logger.info(f'Loaded {all_embeddings_cpu_f32.shape[0]} embeddings from {self.index_dir} onto CPU as float32.')

        dimension = all_embeddings_cpu_f32.shape[1]

        self.faiss_gpu_indexes: List[faiss.GpuIndexFlatIP] = []
        self.faiss_shard_global_offsets: List[int] = []
        self.faiss_cpu_index = None

        num_embeddings_total = all_embeddings_cpu_f32.shape[0]

        # --- MODIFICATION FOR FAISS INDEX DISTRIBUTION ---
        if num_selected_gpus > 0 and num_embeddings_total > 0:
            embeddings_per_gpu = (num_embeddings_total + num_selected_gpus - 1) // num_selected_gpus 

            for i, gpu_id_for_faiss in enumerate(self.gpu_ids_to_use): 
                start_idx = i * embeddings_per_gpu
                end_idx = min((i + 1) * embeddings_per_gpu, num_embeddings_total)
                
                if start_idx >= end_idx:
                    logger.warning(f"GPU {gpu_id_for_faiss} received an empty embedding shard due to distribution (start_idx {start_idx} >= end_idx {end_idx}). Skipping Faiss index creation for this GPU.")
                    continue

                shard_embeddings_tensor = all_embeddings_cpu_f32[start_idx:end_idx]
                shard_embeddings_np = shard_embeddings_tensor.numpy()

                if shard_embeddings_np.shape[0] == 0:
                    logger.warning(f"GPU {gpu_id_for_faiss} has an empty embedding shard (shape 0) after slicing. Skipping Faiss index creation.")
                    continue

                logger.info(f"Creating Faiss GpuIndexFlatIP on GPU {gpu_id_for_faiss} for shard of size {shard_embeddings_np.shape[0]}x{dimension}")
                try:
                    res = faiss.StandardGpuResources()
                    cpu_index = faiss.IndexFlatIP(dimension)
                    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id_for_faiss, cpu_index) 
                    gpu_index.add(shard_embeddings_np)
                    self.faiss_gpu_indexes.append(gpu_index)
                    self.faiss_shard_global_offsets.append(start_idx)
                except Exception as e:
                    logger.error(f"Failed to create Faiss index on GPU {gpu_id_for_faiss}: {e}")
        # --- MODIFICATION FOR FAISS INDEX DISTRIBUTION END ---

        if not self.faiss_gpu_indexes and num_embeddings_total > 0:
            logger.info("No GPU Faiss indexes were created (or no GPUs were selected/available). Attempting to create a single Faiss CPU index.")
            try:
                cpu_index = faiss.IndexFlatIP(dimension)
                cpu_index.add(all_embeddings_cpu_f32.numpy())
                self.faiss_cpu_index = cpu_index
                self.faiss_shard_global_offsets = [0]
                logger.info(f"Successfully created Faiss CPU index with {all_embeddings_cpu_f32.shape[0]} embeddings.")
            except Exception as e:
                logger.error(f"Failed to create Faiss CPU index: {e}")
                self.faiss_cpu_index = None
        elif num_embeddings_total == 0:
            logger.warning("No embeddings loaded. No Faiss index will be created.")


        if not self.faiss_gpu_indexes and self.faiss_cpu_index is None:
            message = "No Faiss GPU or CPU indexes were successfully created. Search will not function."
            if num_embeddings_total == 0:
                message += " This is because no embeddings were loaded."
            else:
                message += " This might be due to issues with GPU availability/selection or Faiss setup."
            logger.error(message)


        self.corpus: Optional[List[Dict]] = None
        if self.verbose:
            logger.info("Verbose mode active. Loading corpus data into memory...")
            self.corpus = load_corpus()
            if self.corpus:
                logger.info(f"Corpus data with {len(self.corpus)} items fully loaded into memory.")
            else:
                logger.warning("Corpus loading returned no data or failed. Verbose output for document content will be unavailable.")
        else:
            logger.info("Verbose mode is False. Corpus data will not be loaded into memory.")


    @torch.no_grad()
    def batch_search(self, queries: List[str], k: int, **kwargs) -> List[List[Dict]]:
        """
        批量检索Top-K文档
        
        功能：这是REAP框架中FE模块的核心检索函数，对应论文公式1：D_t = Retriever(q_t; C)
        
        流程：
        1. 使用e5-large-v2编码器将查询编码为向量
        2. 通过Faiss进行Top-K向量相似度搜索
        3. 返回与每个查询最相关的Top-K文档列表
        
        Args:
            queries: 查询文本列表（通常是子任务的查询q_t）
            k: 返回的Top-K文档数量（论文中设为5）
            **kwargs: 其他可选参数
            
        Returns:
            List[List[Dict]]: 每个查询对应的Top-K文档列表
                每个文档包含：
                - 'doc_id' (int): 文档ID
                - 'score' (float): 相似度分数
                - 'contents' (str): 文档内容（如果verbose=True）
                - 'title' (str): 文档标题（如果有）
        """
        if not self.faiss_gpu_indexes and self.faiss_cpu_index is None:
            logger.warning("No Faiss indexes available for search. Returning empty results.")
            return [[] for _ in queries]

        query_embed_pt: torch.Tensor = self.encoder.encode_queries(queries).float()
        batch_sorted_score, batch_sorted_indices = self._compute_topk(query_embed_pt, k=k)

        results_list: List[List[Dict]] = []
        for query_idx in range(len(queries)):
            results: List[Dict] = []
            for score_val, idx_val in zip(batch_sorted_score[query_idx], batch_sorted_indices[query_idx]):
                doc_id = int(idx_val.item())
                
                if doc_id == -1:
                    continue 
                
                result_item = {
                    'doc_id': doc_id,
                    'score': score_val.item(),
                }

                if self.verbose and self.corpus is not None:
                    if 0 <= doc_id < len(self.corpus):
                        result_item.update(self.corpus[doc_id]) 
                    else:
                        logger.warning(f"doc_id {doc_id} out of bounds for in-memory corpus (size {len(self.corpus)}). Corpus data for this doc will be missing.")
                elif self.verbose and self.corpus is None:
                     logger.warning("Verbose mode is True, but corpus was not loaded or failed to load. Skipping corpus data enrichment.")
                
                results.append(result_item)
            results_list.append(results)
        return results_list

    def _compute_topk(self, query_embed_pt: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        query_embed_np = query_embed_pt.cpu().numpy()
        num_queries = query_embed_np.shape[0]

        if not self.faiss_gpu_indexes and self.faiss_cpu_index is None:
            logger.warning("No Faiss index available in _compute_topk.")
            return torch.empty(num_queries, 0, device='cpu'), torch.empty(num_queries, 0, dtype=torch.long, device='cpu')

        batch_scores_list_np: List[np.ndarray] = []
        batch_indices_list_np: List[np.ndarray] = []

        if self.faiss_gpu_indexes:
            k_per_shard = k 
            for i, faiss_idx in enumerate(self.faiss_gpu_indexes):
                shard_global_offset = self.faiss_shard_global_offsets[i]
                
                D_shard_np, I_shard_local_np = faiss_idx.search(query_embed_np, k_per_shard)
                
                valid_mask = (I_shard_local_np != -1)
                I_shard_global_np = np.where(valid_mask, I_shard_local_np + shard_global_offset, -1)

                batch_scores_list_np.append(D_shard_np)
                batch_indices_list_np.append(I_shard_global_np)
        
        elif self.faiss_cpu_index is not None:
            logger.debug("Performing search on CPU Faiss index.")
            D_np, I_np = self.faiss_cpu_index.search(query_embed_np, k)
            batch_scores_list_np.append(D_np)
            batch_indices_list_np.append(I_np)
        else:
            logger.error("Faiss index logic error in _compute_topk: No valid index found despite initial checks.")
            return torch.empty(num_queries, 0, device='cpu'), torch.empty(num_queries, 0, dtype=torch.long, device='cpu')

        if not batch_scores_list_np:
             logger.warning("No scores/indices collected from Faiss search process.")
             return torch.empty(num_queries, 0, device='cpu'), torch.empty(num_queries, 0, dtype=torch.long, device='cpu')

        batch_scores_all_shards_np = np.concatenate(batch_scores_list_np, axis=1)
        batch_indices_all_shards_np = np.concatenate(batch_indices_list_np, axis=1)

        if batch_scores_all_shards_np.shape[1] == 0:
            logger.warning("Concatenated search results from shards are empty.")
            return torch.empty(num_queries, 0, device='cpu'), torch.empty(num_queries, 0, dtype=torch.long, device='cpu')

        batch_scores_all_shards_pt = torch.from_numpy(batch_scores_all_shards_np).to(query_embed_pt.device)
        batch_indices_all_shards_pt = torch.from_numpy(batch_indices_all_shards_np).long().to(query_embed_pt.device)

        scores_for_topk = torch.full_like(batch_scores_all_shards_pt, float('-inf'))
        valid_indices_mask_pt = (batch_indices_all_shards_pt != -1)
        scores_for_topk[valid_indices_mask_pt] = batch_scores_all_shards_pt[valid_indices_mask_pt]
        
        actual_k_for_final_topk = min(k, scores_for_topk.shape[1])
        if actual_k_for_final_topk == 0:
            logger.warning("No valid candidates after aggregating shards for top-k selection (actual_k_for_final_topk is 0).")
            return torch.empty(num_queries, 0, device='cpu'), torch.empty(num_queries, 0, dtype=torch.long, device='cpu')

        final_top_scores_val, top_indices_in_aggregated = torch.topk(
            scores_for_topk, k=actual_k_for_final_topk, dim=-1, largest=True
        )
        
        final_top_original_indices = torch.gather(
            batch_indices_all_shards_pt, dim=1, index=top_indices_in_aggregated
        )
        return final_top_scores_val.cpu(), final_top_original_indices.cpu()
