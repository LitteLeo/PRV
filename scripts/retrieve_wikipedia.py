#!/usr/bin/env python3
"""
四检索器之 DPR / MonoT5 / E5：从 Wikipedia (psgs_w100) 检索，写 question+ctxs JSONL。
抽样由 --sample-size --seed 控制；Contriever 由 retrieve_four_retrievers.sh 调用 DynamicRAG。
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
import tempfile

# PRV 项目根
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)
if BASE not in sys.path:
    sys.path.insert(0, BASE)


def load_passages_tsv(path: str):
    """与 DynamicRAG load_passages 一致：id, text, title (tsv 列为 id, row[1]=text, row[2]=title)。"""
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] == "id":
                continue
            # id, text, title
            ex = {"id": row[0], "title": row[2] if len(row) > 2 else "", "text": row[1] if len(row) > 1 else ""}
            passages.append(ex)
    return passages


def load_questions(questions_jsonl: str, sample_size: int, seed: int):
    with open(questions_jsonl, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    rng = random.Random(seed)
    n = min(sample_size, len(lines))
    sampled = rng.sample(lines, n)
    return [json.loads(l) for l in sampled]


def write_output(entries, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def ctxs_item_from_passage(p):
    """单条 passage 转为 eval_data 的 ctx 格式（title, text）。"""
    return {"title": p.get("title", ""), "text": p.get("text", p.get("contents", ""))}


# ---------- E5 ----------
def retrieve_e5(questions_list, passages, model_dir, passages_path, top_k, e5_index_dir):
    import torch
    import faiss

    if not e5_index_dir or not os.path.isdir(e5_index_dir):
        raise FileNotFoundError(f"E5 index dir required: {e5_index_dir}")

    from search.simple_encoder import SimpleEncoder

    encoder_path = os.path.join(model_dir, "e5-large-v2")
    encoder = SimpleEncoder(model_name_or_path=encoder_path, max_length=512)
    if torch.cuda.is_available():
        encoder.to("cuda:0")

    shard_paths = sorted(glob.glob(os.path.join(e5_index_dir, "*-shard-*.pt")), key=lambda p: int(p.split("-shard-")[1].split(".")[0]))
    if not shard_paths:
        raise FileNotFoundError(f"No *-shard-*.pt in {e5_index_dir}")
    all_emb = torch.cat([torch.load(p, weights_only=True, map_location="cpu") for p in shard_paths], dim=0).float()
    dimension = all_emb.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(all_emb.numpy())

    # 文档文本（与索引顺序一致，需与建索引时一致：通常 tsv 顺序）
    doc_texts = []
    for p in passages:
        t = p.get("text", "")
        if p.get("title"):
            t = p["title"] + " " + t
        doc_texts.append(t)

    results = []
    for item in questions_list:
        q = item.get("question", "")
        q_vec = encoder.encode_queries([q]).float().numpy()
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        scores, indices = index.search(q_vec, min(top_k, len(passages)))
        ctxs = []
        for idx in indices[0]:
            if 0 <= idx < len(passages):
                ctxs.append(ctxs_item_from_passage(passages[idx]))
        out = {**item, "ctxs": ctxs}
        results.append(out)
    return results


# ---------- DPR ----------
def retrieve_dpr(questions_list, passages, model_dir, top_k, dpr_index_dir):
    import torch
    import numpy as np
    import faiss
    from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

    q_path = os.path.join(model_dir, "dpr-question-encoder")
    ctx_path = os.path.join(model_dir, "dpr-ctx-encoder")
    if not os.path.isdir(q_path) or not os.path.isdir(ctx_path):
        raise FileNotFoundError(f"DPR encoders not found: {q_path}, {ctx_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    q_enc = DPRQuestionEncoder.from_pretrained(q_path).to(device).eval()
    ctx_enc = DPRContextEncoder.from_pretrained(ctx_path).to(device).eval()
    try:
        q_tok = DPRQuestionEncoderTokenizer.from_pretrained(q_path)
        ctx_tok = DPRContextEncoderTokenizer.from_pretrained(ctx_path)
    except Exception:
        from transformers import AutoTokenizer
        q_tok = AutoTokenizer.from_pretrained(q_path)
        ctx_tok = AutoTokenizer.from_pretrained(ctx_path)

    # 建索引：用 ctx encoder 编码所有 passage（无缓存则现场编码，较慢）
    index_path = os.path.join(dpr_index_dir, "dpr_ctx.npy") if dpr_index_dir else None
    if index_path and os.path.isfile(index_path):
        all_ctx_emb = np.load(index_path)
    else:
        batch_size = 32
        all_ctx_emb = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            texts = [p.get("title", "") + " " + p.get("text", "") for p in batch]
            inp = ctx_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                emb = ctx_enc(**inp).pooler_output.cpu().numpy()
            all_ctx_emb.append(emb)
        all_ctx_emb = np.vstack(all_ctx_emb)
        if dpr_index_dir:
            os.makedirs(dpr_index_dir, exist_ok=True)
            np.save(index_path, all_ctx_emb)

    dimension = all_ctx_emb.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(all_ctx_emb)
    faiss_index.add(all_ctx_emb)

    results = []
    for item in questions_list:
        q = item.get("question", "")
        inp = q_tok([q], return_tensors="pt", padding=True, truncation=True, max_length=256)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            q_emb = q_enc(**inp).pooler_output.cpu().numpy()
        faiss.normalize_L2(q_emb)
        scores, indices = faiss_index.search(q_emb, min(top_k, len(passages)))
        ctxs = [ctxs_item_from_passage(passages[int(i)]) for i in indices[0] if 0 <= int(i) < len(passages)]
        results.append({**item, "ctxs": ctxs})
    return results


# ---------- MonoT5：Contriever Top-100 + MonoT5 重排到 top_k ----------
def retrieve_monot5(questions_list, model_dir, top_k, dynamicrag_dir, passages, contriever_embeddings, tmp_dir):
    """先调用 DynamicRAG Contriever 检索 100 条，再用 MonoT5 重排取 top_k。"""
    if not dynamicrag_dir or not os.path.isdir(dynamicrag_dir):
        raise FileNotFoundError("DYNAMICRAG_DIR required for MonoT5 (Contriever 100 + rerank)")

    tmp_query = os.path.join(tmp_dir, "monot5_query.jsonl")
    tmp_100 = os.path.join(tmp_dir, "monot5_contriever_100.jsonl")
    with open(tmp_query, "w", encoding="utf-8") as f:
        for item in questions_list:
            o = {**item, "retrieved_question": item.get("question", "")}
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(
        [
            sys.executable, "retriever.py",
            "--model_name_or_path", os.path.join(model_dir, "contriever"),
            "--passages", passages,
            "--passages_embeddings", contriever_embeddings,
            "--query", tmp_query,
            "--output_dir", tmp_100,
            "--n_docs", "100",
            "--save_or_load_index",
            "--projection_size", "768", "--n_subquantizers", "0", "--n_bits", "8",
        ],
        cwd=dynamicrag_dir,
        env=env,
        check=True,
    )
    with open(tmp_100, "r", encoding="utf-8") as f:
        items_100 = [json.loads(l) for l in f if l.strip()]

    # MonoT5 重排：对每条 (question, ctxs_100) 打分，取 top_k
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        raise ImportError("transformers required for MonoT5 rerank")

    mono_path = os.path.join(model_dir, "monot5-base-msmarco")
    if not os.path.isdir(mono_path):
        raise FileNotFoundError(f"MonoT5 not found: {mono_path}")
    tokenizer = T5Tokenizer.from_pretrained(mono_path)
    model = T5ForConditionalGeneration.from_pretrained(mono_path)
    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    model = model.to(device).eval()

    results = []
    for item in items_100:
        q = item.get("question", item.get("retrieved_question", ""))
        ctxs_raw = item.get("ctxs", [])
        if not ctxs_raw:
            results.append({**item, "ctxs": []})
            continue
        # MonoT5 输入格式: "Query: q Document: doc Relevant:"
        docs = [c.get("text", c.get("contents", "")) for c in ctxs_raw]
        inputs = [f"Query: {q} Document: {d} Relevant:" for d in docs]
        scores_list = []
        batch_size = 8
        torch = __import__("torch")
        true_id = tokenizer.encode("true", add_special_tokens=False)
        true_id = true_id[0] if true_id else 1
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                logits = model(**inp).logits[:, -1, :]
                s = logits[:, true_id].cpu().float().numpy()
            scores_list.extend(s.tolist())
        sorted_idx = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)[:top_k]
        ctxs = [ctxs_raw[j] if "title" in ctxs_raw[j] or "text" in ctxs_raw[j] else ctxs_item_from_passage(ctxs_raw[j]) for j in sorted_idx]
        results.append({**item, "ctxs": ctxs})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", required=True, choices=["dpr", "monot5", "e5"])
    ap.add_argument("--benchmark", required=True, choices=["nq", "hotpotqa", "asqa"])
    ap.add_argument("--questions-jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/home/lfy/projects/models")
    ap.add_argument("--passages", default="/home/lfy/data/wikipedia_embeddings/psgs_w100.tsv")
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--e5-index-dir", default="", help="E5 index dir (e.g. wikipedia e5 shards)")
    ap.add_argument("--dpr-index-dir", default="", help="DPR ctx embeddings cache dir")
    ap.add_argument("--contriever-embeddings", default="", help="For MonoT5: Contriever passages_embeddings glob")
    ap.add_argument("--dynamicrag-dir", default="")
    args = ap.parse_args()

    questions_list = load_questions(args.questions_jsonl, args.sample_size, args.seed)
    if not questions_list:
        print("No questions loaded.", file=sys.stderr)
        return

    passages = load_passages_tsv(args.passages)
    if not passages:
        raise FileNotFoundError(f"No passages from {args.passages}")

    if args.retriever == "e5":
        results = retrieve_e5(questions_list, passages, args.model_dir, args.passages, args.top_k, args.e5_index_dir or None)
    elif args.retriever == "dpr":
        results = retrieve_dpr(questions_list, passages, args.model_dir, args.top_k, args.dpr_index_dir or None)
    elif args.retriever == "monot5":
        dg = args.dynamicrag_dir or (os.path.join(BASE, "..", "DynamicRAG") if BASE else "")
        emb = args.contriever_embeddings or "/home/lfy/data/wikipedia_embeddings/wikipedia_embeddings/passages_*"
        tmp_dir = tempfile.mkdtemp()
        try:
            results = retrieve_monot5(questions_list, args.model_dir, args.top_k, dg, args.passages, emb, tmp_dir)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        raise ValueError(args.retriever)

    write_output(results, args.output)
    print(f"Wrote {len(results)} items to {args.output}")


if __name__ == "__main__":
    main()
