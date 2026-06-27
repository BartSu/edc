"""Generate a declarative statement for every triplet in a canon_kg.txt file.

Uses vLLM offline batch inference with mistralai/Mistral-7B-Instruct-v0.2.
One JSONL line per triplet: {"triplet": [s, p, o], "text": ...}.

Example:
    python datasets/triplet_to_text.py \
        --kg output/tofu_self_canonicalization/iter0/canon_kg.txt \
        --out datasets/datasets/tofu_forget10_kg_iter0_gentext.jsonl
"""
import argparse
import ast
import json
import re

from vllm import LLM, SamplingParams

SYSTEM = (
    "You turn a knowledge-graph triplet (subject, relation, object) into one "
    "natural-language declarative sentence that states the fact in the triplet. "
    'Reply ONLY as JSON: {"text": "..."}'
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


def parse_text(text):
    # Take the last {...} block in case the model emits any preamble.
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for blob in reversed(matches):
        try:
            obj = json.loads(blob)
            if "text" in obj:
                return str(obj["text"]).strip()
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="output/tofu_self_canonicalization/iter0/canon_kg.txt")
    ap.add_argument("--out", default="datasets/datasets/tofu_forget10_kg_iter0_gentext.jsonl")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    triplets = load_triplets(args.kg)
    print(f"Loaded {len(triplets)} triplets from {args.kg}")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
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
            raw = out.outputs[0].text
            text = parse_text(raw)
            if text is None:
                text = ""
            else:
                n_ok += 1
            g.write(json.dumps({"triplet": trip, "text": text}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(triplets)} rows -> {args.out} ({n_ok} parsed OK)")


if __name__ == "__main__":
    main()
