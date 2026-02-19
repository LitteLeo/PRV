import argparse
import pprint
import re
import string
import json
import sys
import os
import csv
from rouge import Rouge

from collections import Counter

sys.setrecursionlimit(3000)

def load_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

def _normalize_guess_records(guess_records):
    """
    Normalize prediction records to standard format.
    Standard format: {"id": ..., "output": [{"answer": ...}], ...}
    Compatible format: {"id": ..., "predicted_answer": ..., "judgement": ..., ...}
    """
    normalized_records = []
    if not guess_records:
        return []

    # Check format by examining key fields in first record
    first_item = guess_records[0]
    if "predicted_answer" in first_item:
        # This is output format from llm_evaluate_predictions.py, needs conversion
        print("INFO: Detected LLM evaluation file format, normalizing...")
        for item in guess_records:
            normalized_item = {
                "id": item.get("id"),
                "question": item.get("question"),
                # Core conversion: put 'predicted_answer' into 'output' structure
                "output": [{"answer": item.get("predicted_answer", "")}]
            }
            # Keep 'judgement' field for subsequent evaluation
            if "judgement" in item:
                normalized_item["judgement"] = item["judgement"]
            
            normalized_records.append(normalized_item)
        return normalized_records
    else:
        # Already standard format or unknown format, return directly
        print("INFO: Detected standard prediction file format.")
        return guess_records


# utility to get gold answers
def get_gold_answers(gold):
    ground_truths = set()
    for item in gold["output"]:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
    return ground_truths


# utility to get max
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    if not scores_for_ground_truths:
        return 0
    return max(scores_for_ground_truths)


# answer normalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# F1 score definition
def _f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # Consistent special answer handling with metrics.py
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# EM score definition
def _exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# Cover EM score definition
def _cover_em_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if not normalized_prediction or not normalized_ground_truth:
        return 0.0
    return float(normalized_ground_truth in normalized_prediction)


# ROUGEL score definition
def _rougel_score(prediction, ground_truth):
    # Consistent with metrics.py, add empty string check
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def _calculate_llm_judged_metrics(guess_records):
    """
    Calculate accuracy if prediction records contain 'judgement' field.
    """
    llm_judged_correct_count = 0
    llm_judged_total_count = 0
    has_judgement_field = False

    for item in guess_records:
        if 'judgement' in item:
            has_judgement_field = True
            judgement = item.get('judgement')
            if judgement == 'True':
                llm_judged_correct_count += 1
                llm_judged_total_count += 1
            elif judgement == 'False':
                llm_judged_total_count += 1
    
    if not has_judgement_field:
        return None

    accuracy = (llm_judged_correct_count / llm_judged_total_count) if llm_judged_total_count > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct_count": llm_judged_correct_count,
        "judged_count": llm_judged_total_count,
    }


def _calculate_metrics(gold_records, guess_records):

    assert len(gold_records) == len(
        guess_records
    ), "different size gold: {} guess: {}".format(len(gold_records), len(guess_records))

    total_evaluated_count = 0

    # downstream metrics
    accuracy_sum = 0.0
    normalized_em_sum = 0.0
    cover_em_sum = 0.0
    normalized_f1_sum = 0.0
    rougel_sum = 0.0

    for guess_item, gold_item in zip(guess_records, gold_records):

        # check ids
        assert (
            str(gold_item["id"]).strip() == str(guess_item["id"]).strip()
        ), "Items must have same order with same IDs after validation"

        gold_candidate_answers = get_gold_answers(gold_item)
        if not gold_candidate_answers:
            continue

        total_evaluated_count += 1

        guess_answer = ""
        if guess_item.get("output") and isinstance(guess_item["output"], list) and len(guess_item["output"]) > 0:
            first_output = guess_item["output"][0]
            if "answer" in first_output and isinstance(first_output["answer"], str):
                guess_answer = first_output["answer"].strip()


        # 0. accuracy = strict exact match
        local_accuracy = 0.0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1.0
        accuracy_sum += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, guess_answer, gold_candidate_answers
        )
        normalized_em_sum += local_em

        # 2. cover em
        local_cover_em = _metric_max_over_ground_truths(
            _cover_em_score, guess_answer, gold_candidate_answers
        )
        cover_em_sum += local_cover_em

        # 3. normalized f1
        local_f1 = _metric_max_over_ground_truths(
            _f1_score, guess_answer, gold_candidate_answers
        )
        normalized_f1_sum += local_f1

        # 4. rougel
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, guess_answer, gold_candidate_answers
        )
        rougel_sum += local_rougel


    accuracy_avg, normalized_em_avg, normalized_f1_avg, rougel_avg, cover_em_avg = 0.0, 0.0, 0.0, 0.0, 0.0
    if total_evaluated_count > 0:
        accuracy_avg = accuracy_sum / total_evaluated_count
        normalized_em_avg = normalized_em_sum / total_evaluated_count
        cover_em_avg = cover_em_sum / total_evaluated_count
        normalized_f1_avg = normalized_f1_sum / total_evaluated_count
        rougel_avg = rougel_sum / total_evaluated_count

    return {
        "downstream": {
            "accuracy": accuracy_avg,
            "em": normalized_em_avg,
            "cover_em": cover_em_avg,
            "f1": normalized_f1_avg,
            "rougel": rougel_avg,
            "evaluated_count": total_evaluated_count,
        },
    }


def validate_input(gold_records, guess_records):

    if len(gold_records) != len(guess_records):
        print(
            "INFO: Different sizes for gold: {} and guess: {} records. Alignment will be attempted based on IDs.".format(
                len(gold_records), len(guess_records)
            )
        )

    id2guess_record = {}
    for guess in guess_records:
        guess_id_str = str(guess["id"]).strip()
        if guess_id_str in id2guess_record:
            raise ValueError(f"Prediction IDs should be unique. Duplicate found: {guess_id_str}")
        id2guess_record[guess_id_str] = guess

    aligned_gold_records = []
    aligned_guess_records = []
    
    processed_gold_ids_for_alignment = set() 
    
    for gold_item in gold_records:
        current_gold_id = str(gold_item["id"]).strip()
        if current_gold_id in id2guess_record:
            if current_gold_id not in processed_gold_ids_for_alignment or True:
                aligned_gold_records.append(gold_item)
                aligned_guess_records.append(id2guess_record[current_gold_id])

    if len(aligned_gold_records) == 0 and len(gold_records) > 0 :
        raise ValueError("ERROR: No matching predictions found for any gold records. Check ID consistency.")
    
    if len(aligned_gold_records) < len(gold_records):
        print(f"Warning: Original gold records: {len(gold_records)}, Matched records for evaluation: {len(aligned_gold_records)}")


    return aligned_gold_records, aligned_guess_records


def evaluate(gold_filepath, guess_filepath):
    pp = pprint.PrettyPrinter(indent=4)

    try:
        gold_records = load_data(gold_filepath)
        guess_records = load_data(guess_filepath)
    except FileNotFoundError as fnf_error:
        print(f"Error: File not found - {fnf_error}")
        return None
    except json.JSONDecodeError as json_error:
        print(f"Error decoding JSON from file: {json_error}")
        print("Please ensure files are in valid JSONL format.")
        return None
    except Exception as e:
        print(f"Error loading data using the 'load_data' function: {e}")
        print("Please ensure your data is in the expected JSONL format or modify the data loading part of this script.")
        return None

    if not gold_records:
        print(f"Error: Gold file {gold_filepath} is empty or could not be loaded correctly.")
        return None
    if not guess_records:
        print(f"Error: Guess file {guess_filepath} is empty or could not be loaded correctly.")
        return None

    # Normalize prediction file records
    guess_records = _normalize_guess_records(guess_records)

    # 0. validate input
    try:
        gold_records, guess_records = validate_input(gold_records, guess_records)
    except ValueError as ve:
        print(f"Validation Error: {ve}")
        return None


    if not gold_records or not guess_records:
        print("Error: No records available for evaluation after validation/alignment. Check ID matching and data format.")
        return {
            "downstream": {
                "accuracy": 0.0, "em": 0.0, "cover_em": 0.0, "f1": 0.0, "rougel": 0.0, "evaluated_count": 0,
            }
        }

    # 1. downstream metrics
    result = _calculate_metrics(gold_records, guess_records)

    # 2. LLM-judged metrics (if applicable)
    llm_metrics = _calculate_llm_judged_metrics(guess_records)
    if llm_metrics:
        result["llm_judged_metrics"] = llm_metrics

    pp.pprint(result)
    return result


def evaluate_folder(guess_folder_path, gold_filepath, output_csv_path):
    """
    Evaluate all .jsonl files in specified folder and write key metrics to CSV file.
    """
    if not os.path.isdir(guess_folder_path):
        print(f"Error: The provided path '{guess_folder_path}' is not a valid folder.")
        return

    # Find all .jsonl files in folder
    guess_files = [f for f in os.listdir(guess_folder_path) if f.endswith('.jsonl')]
    if not guess_files:
        print(f"Error: No .jsonl files found in folder '{guess_folder_path}'.")
        return

    print(f"Found {len(guess_files)} prediction files in '{guess_folder_path}'.")
    print(f"Using gold answer file: '{gold_filepath}'")
    print(f"Results will be saved to: '{output_csv_path}'")

    results_data = []

    for filename in sorted(guess_files):
        guess_filepath = os.path.join(guess_folder_path, filename)
        print(f"\n--- Evaluating: {filename} ---")
        
        # Call existing evaluation function
        evaluation_results = evaluate(gold_filepath, guess_filepath)

        if not evaluation_results:
            print(f"Warning: File '{filename}' evaluation failed or returned no results.")
            continue

        # Extract required metrics
        downstream_metrics = evaluation_results.get("downstream", {})
        llm_metrics = evaluation_results.get("llm_judged_metrics", {})

        em_score = downstream_metrics.get("em", "N/A")
        cover_em_score = downstream_metrics.get("cover_em", "N/A")
        f1_score = downstream_metrics.get("f1", "N/A")
        llm_accuracy = llm_metrics.get("accuracy", "N/A") if llm_metrics else "N/A"

        results_data.append({
            "filename": filename,
            "em": em_score,
            "cover_em": cover_em_score,
            "f1": f1_score,
            "llm_judged_accuracy": llm_accuracy
        })

    # Write results to CSV file
    if not results_data:
        print("No valid evaluation results to write to CSV file.")
        return

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["filename", "em", "cover_em", "f1", "llm_judged_accuracy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results_data)
        print(f"\nEvaluation summary successfully saved to '{output_csv_path}'")
    except IOError as e:
        print(f"Error: Cannot write to CSV file '{output_csv_path}'. Reason: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA predictions (accuracy, EM, F1, ROUGE-L).")
    parser.add_argument("gold_file", help="Gold answer file path (JSONL format)")
    parser.add_argument("--guess_file", help="Single prediction file path (JSONL format)")
    parser.add_argument("--guess_folder", help="Folder path containing multiple prediction files")
    parser.add_argument("--output_csv", help="When using --guess_folder, specify output CSV file path", default="evaluation_summary.csv")

    args = parser.parse_args()
    
    if args.guess_file:
        print(f"Evaluating single file '{args.guess_file}' against '{args.gold_file}'...")
        evaluation_results = evaluate(args.gold_file, args.guess_file)
        if evaluation_results:
            downstream_results = evaluation_results.get("downstream", {})
            if downstream_results.get("evaluated_count", 0) == 0:
                print("Evaluation completed, but no valid downstream metric samples evaluated (e.g., due to missing gold answers or ID mismatches).")
            else:
                print("Downstream metrics evaluation completed.")
            
            if "llm_judged_metrics" in evaluation_results:
                 print("LLM-judged metrics evaluation completed.")
        else:
            print("Evaluation failed or produced no results.")

    elif args.guess_folder:
        evaluate_folder(args.guess_folder, args.gold_file, args.output_csv)
    
    else:
        print("Error: You must provide either --guess_file or --guess_folder parameter.")
        parser.print_help()