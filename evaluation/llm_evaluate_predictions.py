import json
import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import traceback
import time

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from rag_pipeline_lib import llm_adapter 

def load_jsonl(filepath: str):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding error when parsing file {filepath}: {e}", file=sys.stderr)
        return None
    return data
    

def evaluate_single_item(item_data, max_retries=3, retry_delay=5):
    prediction, ground_truth_item = item_data
    item_id = prediction.get("id")
    
    for attempt in range(max_retries):
        try:
            question = ground_truth_item.get("input")
            golden_answer = ground_truth_item.get("output", [{}])[0].get("answer", "")
            predicted_answer = prediction.get("output", [{}])[0].get("answer", "")

            if not all([question, golden_answer]):
                judgement = "Skipped: Missing ground truth data."
            else:
                judgement = llm_adapter.evaluate_answer(
                    question=question,
                    golden_answer=golden_answer,
                    predicted_answer=predicted_answer
                )

            # If judgement is not an error, success, break out of retry loop
            if not judgement.startswith("Evaluation Error:"):
                return {
                    "id": item_id,
                    "question": question,
                    "golden_answer": golden_answer,
                    "predicted_answer": predicted_answer,
                    "judgement": judgement
                }
            
            # If it's an error, log and prepare for retry
            error_message = judgement
            print(f"\nID {item_id} evaluation failed (attempt {attempt + 1}/{max_retries}): {error_message}", file=sys.stderr)

        except Exception as e:
            error_message = f"Processing Error: {e}"
            print(f"\nSerious error when processing ID {item_id} (attempt {attempt + 1}/{max_retries}): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        
        # If not the last attempt, wait and retry
        if attempt < max_retries - 1:
            print(f"Will retry after {retry_delay} seconds...", file=sys.stderr)
            time.sleep(retry_delay)

    # After all retries failed, return final error message
    return {
        "id": item_id,
        "question": ground_truth_item.get("input", "N/A"),
        "golden_answer": ground_truth_item.get("output", [{}])[0].get("answer", "N/A"),
        "predicted_answer": prediction.get("output", [{}])[0].get("answer", "N/A"),
        "judgement": f"Failed after {max_retries} attempts: {error_message}"
    }


# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction answers for correctness using LLM.")
    parser.add_argument("--prediction_file", required=True, help="Path to JSONL file containing model predictions.")
    parser.add_argument("--dataset_file", required=True, help="Path to original dataset JSONL file containing ground truth answers.")
    parser.add_argument("--output_file", required=True, help="Path to JSONL file to save evaluation results.")
    parser.add_argument("--max_workers", type=int, default=16, help="Maximum number of worker threads for parallel processing (default 8).")
    args = parser.parse_args()

    # Load data
    print(f"Loading predictions from {args.prediction_file}...")
    predictions = load_jsonl(args.prediction_file)
    if predictions is None: return

    print(f"Loading dataset from {args.dataset_file}...")
    dataset = load_jsonl(args.dataset_file)
    if dataset is None: return

    # Create a mapping from id to dataset entry for fast lookup
    ground_truth_map = {item['id']: item for item in dataset}

    # Prepare task list
    tasks = []
    for pred in predictions:
        item_id = pred.get("id")
        if not item_id:
            print(f"Warning: Missing 'id' in prediction, skipping record: {pred}", file=sys.stderr)
            continue
        if item_id in ground_truth_map:
            tasks.append((pred, ground_truth_map[item_id]))
        else:
            print(f"Warning: ID '{item_id}' not found in dataset, skipping this prediction.", file=sys.stderr)

    total_items = len(tasks)
    if total_items == 0:
        print("No matching items to evaluate. Exiting.")
        return

    print(f"Will evaluate {total_items} records (using up to {args.max_workers} worker threads)...\n")

    # Process in parallel and write results
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Use lambda to pass additional parameters to the function in map
            results_iterator = executor.map(lambda task: evaluate_single_item(task), tasks)
            
            correct_count = 0
            total_judged = 0

            for result in tqdm(results_iterator, total=total_items, desc="Evaluating predictions"):
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                outfile.flush()
                if result.get("judgement") == "True":
                    correct_count += 1
                if result.get("judgement") in ["True", "False"]:
                    total_judged += 1
    
    print(f"\nEvaluation completed, results saved to: {args.output_file}")
    if total_judged > 0:
        accuracy = (correct_count / total_judged) * 100
        print(f"\n--- Evaluation Summary ---")
        print(f"Number of correct judgements: {correct_count}")
        print(f"Total number of valid judgements: {total_judged}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"-------------------------")