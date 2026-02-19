#!/usr/bin/env python3
"""
评估脚本：将预测结果与原始答案进行比较

用法：
    python scripts/evaluate_with_answers.py \
        --original_file /home/lfy/data/eval_data/eli5.jsonl \
        --predictions_file /home/lfy/projects/REAP/experiments/eli5_predictions.jsonl \
        --output_file /home/lfy/projects/REAP/experiments/eli5_metrics.txt
"""

import argparse
import json
import sys
from pathlib import Path

# 添加评估脚本路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'evaluation'))
from evaluation_script import normalize_answer, _f1_score, _exact_match_score


def load_original_answers(original_file):
    """从原始文件加载答案"""
    answers_map = {}
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            item_id = item.get('id', '')
            # 提取答案列表
            answers = item.get('answers', [])
            # 如果 answers 是列表，提取文本
            answer_texts = []
            for ans in answers:
                if isinstance(ans, str):
                    answer_texts.append(ans)
                elif isinstance(ans, dict):
                    answer_texts.append(ans.get('text', ans.get('answer', '')))
                else:
                    answer_texts.append(str(ans))
            answers_map[item_id] = [a.strip() for a in answer_texts if a.strip()]
    return answers_map


def evaluate_predictions(original_file, predictions_file):
    """评估预测结果"""
    # 加载原始答案
    answers_map = load_original_answers(original_file)
    
    # 加载预测结果
    predictions = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            predictions.append(item)
    
    # 计算指标
    total = 0
    em_correct = 0
    f1_scores = []
    
    for pred in predictions:
        item_id = str(pred.get('id', ''))
        pred_answer = pred.get('output', [{}])[0].get('answer', '') if pred.get('output') else ''
        
        if not pred_answer or item_id not in answers_map:
            continue
        
        gold_answers = answers_map[item_id]
        if not gold_answers:
            continue
        
        total += 1
        
        # EM
        em = max(_exact_match_score(pred_answer, gold) for gold in gold_answers)
        em_correct += em
        
        # F1
        f1 = max(_f1_score(pred_answer, gold) for gold in gold_answers)
        f1_scores.append(f1)
    
    em_score = em_correct / total if total > 0 else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    return {
        'total': total,
        'em': em_score,
        'f1': avg_f1,
        'em_count': em_correct,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate REAP predictions')
    parser.add_argument('--original_file', type=str, required=True,
                        help='Original dataset file with answers')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='REAP predictions file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output metrics file')
    
    args = parser.parse_args()
    
    metrics = evaluate_predictions(args.original_file, args.predictions_file)
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total: {metrics['total']}\n")
        f.write(f"EM: {metrics['em']:.4f} ({metrics['em_count']}/{metrics['total']})\n")
        f.write(f"F1: {metrics['f1']:.4f}\n")
    
    print(f"Results saved to {args.output_file}")
    print(f"EM: {metrics['em']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
