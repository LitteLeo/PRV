#!/usr/bin/env python3
"""
为 REAP 框架生成向量索引的脚本

用法：
    python scripts/generate_index.py \
        --corpus_file /path/to/your_corpus.jsonl \
        --output_dir /home/lfy/data/corpus_embeddings \
        --model_path /home/lfy/projects/models/e5-large-v2 \
        --shard_size 50000
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from search.simple_encoder import SimpleEncoder
from logger_config import logger


def load_corpus_from_jsonl(filepath: str) -> List[Dict]:
    """从 JSONL 文件加载语料库"""
    corpus = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                # 假设格式为 {"id": ..., "text": ..., "title": ...} 或类似
                # 根据你的实际格式调整
                if 'text' in item:
                    corpus.append({
                        'id': item.get('id', line_num),
                        'contents': item.get('text', item.get('content', '')),
                        'title': item.get('title', ''),
                    })
                elif 'contents' in item:
                    corpus.append(item)
                else:
                    # 尝试其他常见字段
                    corpus.append({
                        'id': item.get('id', line_num),
                        'contents': str(item),
                        'title': item.get('title', ''),
                    })
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num} parse error: {e}")
                continue
    return corpus


def encode_corpus(
    corpus: List[Dict],
    encoder: SimpleEncoder,
    shard_size: int = 50000,
    output_dir: str = './embeddings'
) -> None:
    """编码语料库并保存为 shard 文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    total_docs = len(corpus)
    num_shards = (total_docs + shard_size - 1) // shard_size
    
    logger.info(f"Encoding {total_docs} documents into {num_shards} shards...")
    
    # 准备文档文本（用于编码）
    # E5 模型对 passage 使用 "passage: " 前缀
    doc_texts = []
    for doc in corpus:
        text = doc.get('contents', '')
        if 'title' in doc and doc['title']:
            text = f"{doc['title']}\n{text}"
        doc_texts.append(f"passage: {text}")
    
    # 分片编码
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, total_docs)
        
        shard_texts = doc_texts[start_idx:end_idx]
        logger.info(f"Encoding shard {shard_idx + 1}/{num_shards} (docs {start_idx}-{end_idx})...")
        
        # 编码
        embeddings = encoder._do_encode(shard_texts, batch_size=128)
        
        # 保存 shard
        shard_path = os.path.join(output_dir, f'corpus-shard-{shard_idx}.pt')
        torch.save(embeddings, shard_path)
        logger.info(f"Saved shard {shard_idx + 1} to {shard_path} (shape: {embeddings.shape})")
    
    logger.info(f"All embeddings saved to {output_dir}")


def save_corpus_data(corpus: List[Dict], output_path: str):
    """保存语料库数据（用于返回文档内容）"""
    import pickle
    
    # 保存为 pickle（更快）
    with open(output_path, 'wb') as f:
        pickle.dump(corpus, f)
    logger.info(f"Corpus data saved to {output_path} ({len(corpus)} documents)")


def main():
    parser = argparse.ArgumentParser(description='Generate vector index for REAP framework')
    parser.add_argument('--corpus_file', type=str, required=True,
                        help='Input corpus file (JSONL format)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for embedding shards')
    parser.add_argument('--corpus_data_file', type=str, default=None,
                        help='Output file for corpus data (pickle format). If not set, will be {output_dir}/../corpus.pkl')
    parser.add_argument('--model_path', type=str, 
                        default='/home/lfy/projects/models/e5-large-v2',
                        help='Path to E5 model')
    parser.add_argument('--shard_size', type=int, default=50000,
                        help='Number of documents per shard')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for encoding (cuda:0 or cpu)')
    
    args = parser.parse_args()
    
    # 加载语料库
    logger.info(f"Loading corpus from {args.corpus_file}...")
    corpus = load_corpus_from_jsonl(args.corpus_file)
    logger.info(f"Loaded {len(corpus)} documents")
    
    # 初始化编码器
    logger.info(f"Loading encoder from {args.model_path}...")
    encoder = SimpleEncoder(
        model_name_or_path=args.model_path,
        prefix_type='query_or_passage',
        pool_type='avg',
        max_length=512
    )
    encoder.to(args.device)
    encoder.eval()
    
    # 编码并保存
    encode_corpus(corpus, encoder, args.shard_size, args.output_dir)
    
    # 保存语料库数据
    corpus_data_file = args.corpus_data_file
    if corpus_data_file is None:
        corpus_data_file = os.path.join(os.path.dirname(args.output_dir), 'corpus.pkl')
    save_corpus_data(corpus, corpus_data_file)
    
    logger.info("Index generation completed!")


if __name__ == '__main__':
    main()
