#!/usr/bin/env python3
"""
更新配置文件以适配当前数据集
"""
import argparse
import re
import os

def update_config(dataset_name, embeddings_dir, corpus_file):
    config_file = "/home/lfy/projects/REAP/config.py"
    
    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 更新检索服务配置
    # 更新 start_e5_server_main.py
    server_file = "/home/lfy/projects/REAP/search/start_e5_server_main.py"
    with open(server_file, 'r', encoding='utf-8') as f:
        server_content = f.read()
    
    # 替换索引目录
    server_content = re.sub(
        r"index_dir=['\"].*?['\"]",
        f"index_dir='{embeddings_dir}'",
        server_content
    )
    
    # 替换模型路径
    server_content = re.sub(
        r"model_name_or_path=['\"].*?['\"]",
        f"model_name_or_path='/home/lfy/projects/models/e5-large-v2'",
        server_content
    )
    
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(server_content)
    
    # 更新 e5_searcher.py 中的语料库路径
    searcher_file = "/home/lfy/projects/REAP/search/e5_searcher.py"
    with open(searcher_file, 'r', encoding='utf-8') as f:
        searcher_content = f.read()
    
    # 替换语料库路径
    searcher_content = re.sub(
        r"pickle_path = ['\"].*?['\"]",
        f"pickle_path = '{corpus_file}'",
        searcher_content
    )
    
    searcher_content = re.sub(
        r"local_path = ['\"].*?['\"]",
        f"local_path = '{os.path.dirname(corpus_file)}/'",
        searcher_content
    )
    
    with open(searcher_file, 'w', encoding='utf-8') as f:
        f.write(searcher_content)
    
    print(f"Updated config for {dataset_name}")
    print(f"  - Embeddings: {embeddings_dir}")
    print(f"  - Corpus: {corpus_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--embeddings_dir', required=True)
    parser.add_argument('--corpus_file', required=True)
    args = parser.parse_args()
    
    update_config(args.dataset_name, args.embeddings_dir, args.corpus_file)
