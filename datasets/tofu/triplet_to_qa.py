"""Generate a (question, answer) pair for every triplet in a canon_kg.txt file.

Uses vLLM offline batch inference with mistralai/Mistral-7B-Instruct-v0.2.
One JSONL line per triplet: {"triplet": [s, p, o], "question": ..., "answer": ...}.

Example:
    python datasets/triplet_to_qa.py \
        --kg output/tofu_self_canonicalization/iter0/canon_kg.txt \
        --out datasets/datasets/tofu_forget10_kg_iter0_genqa.jsonl
"""
import argparse
import ast
import json
import re

from vllm import LLM, SamplingParams

SYSTEM = (
    "You turn a knowledge-graph triplet (subject, relation, object) into one "
    "natural-language question and its answer. The answer must state the fact in "
    'the triplet. Reply ONLY as JSON: {"question": "...", "answer": "..."}'
)


def load_triplets(path):
    triplets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items = ast.literal_eval(line)
            except Exception:
                continue
            for t in items:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    triplets.append([str(x).strip() for x in t[:3]])
    return triplets


def parse_qa(text):
    # Take the last {...} block in case the model emits any preamble.
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for blob in reversed(matches):
        try:
            obj = json.loads(blob)
            if "question" in obj and "answer" in obj:
                return str(obj["question"]).strip(), str(obj["answer"]).strip()
        except Exception:
            continue
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="output/tofu_self_canonicalization/iter0/canon_kg.txt")
    ap.add_argument("--out", default="datasets/datasets/tofu_forget10_kg_iter0_genqa.jsonl")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    triplets = load_triplets(args.kg)
    print(f"Loaded {len(triplets)} triplets from {args.kg}")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # Mistral-7B-Instruct-v0.2's chat template has no system role, so fold the
    # instructions into the user turn.
    conversations = [
        [
            {"role": "user", "content": f"{SYSTEM}\n\nTriplet: ({s}, {p}, {o})"},
        ]
        for s, p, o in triplets
    ]

    outputs = llm.chat(conversations, sampling)

    n_ok = 0
    with open(args.out, "w") as g:
        for trip, out in zip(triplets, outputs):
            text = out.outputs[0].text
            q, a = parse_qa(text)
            if q is None:
                q, a = "", ""
            else:
                n_ok += 1
            g.write(json.dumps({"triplet": trip, "question": q, "answer": a}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(triplets)} rows -> {args.out} ({n_ok} parsed OK)")


if __name__ == "__main__":
    main()
