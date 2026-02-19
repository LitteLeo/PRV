#!/usr/bin/env python
import json

src = "/home/lfy/data/eval_data/2wikimqa.jsonl"
dst = "/home/lfy/data/eval_data/2wikimqa_gold_reap_format.jsonl"

def extract_answers(ex):
    """从原始样本里提取答案列表，统一成 [{'answer': '...'}, ...]"""
    answers = []

    if "answers" in ex:
        vals = ex["answers"]
        if isinstance(vals, list):
            for a in vals:
                if isinstance(a, str):
                    if a.strip():
                        answers.append({"answer": a.strip()})
                elif isinstance(a, dict):
                    txt = a.get("text") or a.get("answer") or ""
                    if txt and txt.strip():
                        answers.append({"answer": txt.strip()})
        elif isinstance(vals, str):
            if vals.strip():
                answers.append({"answer": vals.strip()})
    elif "answer" in ex:
        txt = ex["answer"]
        if isinstance(txt, str) and txt.strip():
            answers.append({"answer": txt.strip()})

    return answers

cnt_in, cnt_out = 0, 0
with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        cnt_in += 1
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        qid = ex.get("id")
        question = ex.get("question") or ex.get("input") or ""

        answers = extract_answers(ex)
        if not answers:
            continue  # 没答案的样本跳过

        out = {
            "id": str(qid),
            "input": question,
            "output": answers,
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        cnt_out += 1

print(f"Converted {cnt_out}/{cnt_in} records to REAP gold format.")
print(f"Saved to: {dst}")
