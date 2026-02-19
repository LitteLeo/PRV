import json
import argparse
import sys
import os
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import traceback

# --- Path setup ---
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import project modules ---
import config
from rag_pipeline_lib import llm_adapter
from rag_pipeline_lib.pipeline import run_multistep_pipeline, Tracer, tracer_context


# --- Data loading and processing functions ---
def load_input_dataset(filepath: str):
    """Load input dataset from JSONL file."""
    dataset = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if "id" not in data or "input" not in data:
                        print(f"Warning: Line {line_number} skipped, missing 'id' or 'input' field.", file=sys.stderr)
                        continue
                    dataset.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Line {line_number} skipped, unable to parse JSON.", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Input file {filepath} not found.", file=sys.stderr)
        return None
    return dataset

def process_single_item(item):
    """Process a single data item: call multi-step RAG and format results."""
    item_id = item["id"]
    query = item["input"]

    # 创建追踪器来记录LLM调用
    tracer = Tracer()
    tracer_context.set(tracer)

    try:
        # Call core pipeline function without detailed logging
        predicted_answer = run_multistep_pipeline(query, verbose=False, trace_collector=tracer)

        # 统计LLM调用信息
        llm_call_count = len(tracer.log)  # 总LLM调用次数
        iteration_count = tracer.iteration_count  # 迭代次数
        
        # 统计不同类型的LLM调用
        call_types = {}
        for log_entry in tracer.log:
            func_name = log_entry.get("adapter_function_name", "unknown")
            call_types[func_name] = call_types.get(func_name, 0) + 1
    except Exception as e:
        print(f"\nError: Uncaught exception occurred during pipeline execution for ID {item_id}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        predicted_answer = f"Error: Serious error occurred while generating answer - {e}"
        llm_call_count = 0
        iteration_count = 0
        call_types = {}

    return {
        "id": str(item_id),
        "question": query,
        "output": [{"answer": predicted_answer}],
        # 添加LLM调用统计信息
        "llm_stats": {
            "total_llm_calls": llm_call_count,
            "iterations": iteration_count,
            "call_breakdown": call_types,  # 各类型调用次数
            "avg_calls_per_iteration": llm_call_count / max(iteration_count, 1) if iteration_count > 0 else 0
        }
    }


# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Read questions from dataset, generate predictions using multi-step RAG pipeline, and save results.")
    parser.add_argument("input_file", help="Path to input dataset file (JSONL format, containing 'id' and 'input').")
    parser.add_argument("output_file", help="Path to output prediction file (JSONL format).")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of worker threads for parallel processing (default 8).")
    parser.add_argument("--sample_size", type=int, default=None, help="Randomly sample specified number of samples for processing. If not provided, process all samples.")
    parser.add_argument("--sequential_sampling", action="store_true", help="If --sample_size is set, sample sequentially from the beginning of the dataset instead of randomly.")
    parser.add_argument("--resume", action="store_true", help="If enabled, continue processing from existing results in output file instead of overwriting.")
    args = parser.parse_args()

    print("--- Configuring and initializing LLM provider... ---")
    try:
        llm_adapter.configure_llm_provider()
    except Exception as e:
        print(f"Error: Unable to configure LLM provider: {e}", file=sys.stderr)
        sys.exit(1)
    print("--- Initialization complete. ---\n")

    print(f"Loading input dataset from {args.input_file}...")
    input_data = load_input_dataset(args.input_file)

    if not input_data:
        print("Failed to load input data, exiting program.")
        return

    # --- Sampling logic (moved before resume logic) ---
    if args.sample_size is not None and args.sample_size > 0:
        if args.sample_size < len(input_data):
            # Ensure random seed is set before sampling for reproducibility
            if not args.sequential_sampling:
                random.seed(61) # for reproducibility default 42
                print(f"Randomly sampling {args.sample_size} records from {len(input_data)} records...")
                input_data = random.sample(input_data, args.sample_size)
            else:
                print(f"Sequentially sampling first {args.sample_size} records from {len(input_data)} records...")
                input_data = input_data[:args.sample_size]
        else:
            print(f"Sample size ({args.sample_size}) is greater than or equal to dataset size ({len(input_data)}). Will process all data.")


    # --- Resume logic (executed after sampling) ---
    open_mode = 'w'
    if args.resume:
        open_mode = 'a'
        if os.path.exists(args.output_file):
            print(f"Resume mode enabled. Reading processed IDs from {args.output_file}...")
            processed_ids = set()
            try:
                with open(args.output_file, 'r', encoding='utf-8') as f_resume:
                    for line in f_resume:
                        try:
                            data = json.loads(line)
                            if 'id' in data:
                                processed_ids.add(str(data['id']))
                        except json.JSONDecodeError:
                            print(f"Warning: Unable to parse line in output file: {line.strip()}", file=sys.stderr)
                
                if processed_ids:
                    original_count = len(input_data)
                    input_data = [item for item in input_data if str(item['id']) not in processed_ids]
                    print(f"Found {len(processed_ids)} processed records. Will process remaining {len(input_data)}/{original_count} records.")

            except Exception as e:
                print(f"Warning: Error reading output file for resume: {e}. Will continue in append mode but duplicates may exist.", file=sys.stderr)
        else:
            print(f"Resume mode enabled, but output file {args.output_file} does not exist. Will create a new file.")


    total_items = len(input_data)
    if total_items == 0:
        print("No new records to process. Exiting program.")
        return

    print(f"Generating predictions for {total_items} records (using up to {args.max_workers} worker threads)...\n")

    with open(args.output_file, open_mode, encoding='utf-8') as outfile:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results_iterator = executor.map(process_single_item, input_data)
            for result in tqdm(results_iterator, total=total_items, desc="Generating predictions"):
                            
                try:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()
                except Exception as e:
                    print(f"\nSerious error occurred while processing and writing results: {e}", file=sys.stderr)

    print(f"\nAll predictions processed and saved to: {args.output_file}")

if __name__ == "__main__":
    main()
