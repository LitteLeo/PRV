#!/usr/bin/env python3
"""
分析推理结果中的LLM调用统计信息
"""
import json
import argparse
import sys
from collections import defaultdict

def analyze_llm_stats(predictions_file):
    """分析LLM调用统计"""
    total_items = 0
    total_llm_calls = 0
    total_iterations = 0
    call_type_counts = defaultdict(int)
    
    items_with_stats = []
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                total_items += 1
                
                if "llm_stats" in item:
                    stats = item["llm_stats"]
                    total_llm_calls += stats.get("total_llm_calls", 0)
                    total_iterations += stats.get("iterations", 0)
                    
                    # 统计各类型调用
                    for call_type, count in stats.get("call_breakdown", {}).items():
                        call_type_counts[call_type] += count
                    
                    items_with_stats.append({
                        "id": item.get("id"),
                        "llm_calls": stats.get("total_llm_calls", 0),
                        "iterations": stats.get("iterations", 0)
                    })
            except json.JSONDecodeError:
                continue
    
    if total_items == 0:
        print("No items found in predictions file")
        return
    
    # 计算平均值
    avg_llm_calls = total_llm_calls / total_items if total_items > 0 else 0
    avg_iterations = total_iterations / total_items if total_items > 0 else 0
    
    # 输出统计结果
    print("=" * 60)
    print("LLM Call Statistics")
    print("=" * 60)
    print(f"Total items processed: {total_items}")
    print(f"\nOverall Statistics:")
    print(f"  Total LLM calls: {total_llm_calls}")
    print(f"  Average LLM calls per query: {avg_llm_calls:.2f}")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Average iterations per query: {avg_iterations:.2f}")
    print(f"  Average LLM calls per iteration: {total_llm_calls / max(total_iterations, 1):.2f}")
    
    print(f"\nCall Type Breakdown:")
    for call_type, count in sorted(call_type_counts.items(), key=lambda x: x[1], reverse=True):
        avg_per_item = count / total_items if total_items > 0 else 0
        print(f"  {call_type}: {count} total, {avg_per_item:.2f} avg per query")
    
    # 分布统计
    if items_with_stats:
        llm_calls_list = [item["llm_calls"] for item in items_with_stats]
        iterations_list = [item["iterations"] for item in items_with_stats]
        
        print(f"\nDistribution:")
        print(f"  LLM calls - Min: {min(llm_calls_list)}, Max: {max(llm_calls_list)}, Median: {sorted(llm_calls_list)[len(llm_calls_list)//2]}")
        print(f"  Iterations - Min: {min(iterations_list)}, Max: {max(iterations_list)}, Median: {sorted(iterations_list)[len(iterations_list)//2]}")
    
    return {
        "total_items": total_items,
        "avg_llm_calls": avg_llm_calls,
        "avg_iterations": avg_iterations,
        "call_type_counts": dict(call_type_counts)
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM call statistics from predictions')
    parser.add_argument('predictions_file', help='Path to predictions JSONL file')
    parser.add_argument('--output', help='Output JSON file for statistics')
    
    args = parser.parse_args()
    
    stats = analyze_llm_stats(args.predictions_file)
    
    if args.output and stats:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStatistics saved to: {args.output}")

if __name__ == '__main__':
    main()
