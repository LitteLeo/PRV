#!/usr/bin/env python3
"""
准备数据集：转换格式 + 提取语料库生成索引

用法：
    python scripts/prepare_dataset.py \
        --input_file /home/lfy/data/eval_data/eli5.jsonl \
        --output_file /home/lfy/data/eval_data/eli5_reap_format.jsonl \
        --corpus_output /home/lfy/data/corpus_eli5.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_corpus_from_ctxs(ctxs, doc_id_prefix=""):
    """从 ctxs 字段提取文档"""
    corpus = []
    seen_texts = set()  # 去重
    
    for idx, ctx in enumerate(ctxs):
        # 根据实际格式调整，常见格式：
        text = ctx.get('text', ctx.get('content', ctx.get('passage', '')))
        title = ctx.get('title', '')
        
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        
        corpus.append({
            'id': f"{doc_id_prefix}_doc_{len(corpus)}",
            'contents': text,
            'title': title,
        })
    
    return corpus


def convert_dataset(input_file, output_file, corpus_output=None):
    """转换数据集格式"""
    converted = []
    all_corpus = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                # 提取问题（优先用 question，否则用 retrieved_question）
                question = item.get('question', item.get('retrieved_question', ''))
                if not question:
                    print(f"Warning: Line {line_num} has no question field, skipping")
                    continue
                
                # 转换为 REAP 格式
                converted_item = {
                    'id': item.get('id', f'item_{line_num}'),
                    'input': question,
                    # 保留原始答案用于后续评估
                    '_original_answers': item.get('answers', []),
                }
                converted.append(converted_item)
                
                # 提取语料库（从 ctxs 字段）
                if corpus_output and 'ctxs' in item:
                    corpus_docs = extract_corpus_from_ctxs(
                        item['ctxs'], 
                        doc_id_prefix=converted_item['id']
                    )
                    all_corpus.extend(corpus_docs)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} parse error: {e}")
                continue
    
    # 保存转换后的数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted:
            # 写入时不包含 _original_answers（仅用于评估）
            output_item = {'id': item['id'], 'input': item['input']}
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted)} items to {output_file}")
    
    # 保存语料库（去重）
    if corpus_output and all_corpus:
        # 按内容去重
        unique_corpus = {}
        for doc in all_corpus:
            text_key = doc['contents']
            if text_key not in unique_corpus:
                unique_corpus[text_key] = doc
        
        with open(corpus_output, 'w', encoding='utf-8') as f:
            for doc in unique_corpus.values():
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Extracted {len(unique_corpus)} unique documents to {corpus_output}")
    
    return converted, len(unique_corpus) if corpus_output else 0


def main():
    parser = argparse.ArgumentParser(description='Convert dataset format for REAP')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input dataset file (your format)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output dataset file (REAP format)')
    parser.add_argument('--corpus_output', type=str, default=None,
                        help='Output corpus file (optional, extracted from ctxs)')
    
    args = parser.parse_args()
    
    convert_dataset(args.input_file, args.output_file, args.corpus_output)


if __name__ == '__main__':
    main()
