"""Convert TOFU (locuslab/TOFU) QA pairs into EDC-style input passages.

EDC expects datasets/{name}.txt with ONE passage per line.
Run from the root of the edc repo so the default --out path lands in ./datasets/.

Examples:
    python tofu.py --config forget10 --mode qa
    python tofu.py --config full --mode chunk --chunk_size 20
"""
import argparse
import json
import os
import re

from huggingface_hub import hf_hub_download


def clean(text: str) -> str:
    # EDC reads one passage per line, so collapse internal newlines/whitespace.
    return re.sub(r"\s+", " ", text).strip()


def load_tofu(config: str):
    """Load a TOFU config as a list of dicts, bypassing the datasets builder.

    Avoids both the schema-cast mismatch and the fsspec/datasets
    LocalFileSystem incompatibility by reading the raw json directly.
    """
    path = hf_hub_download(
        repo_id="locuslab/TOFU",
        filename=f"{config}.json",
        repo_type="dataset",
    )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)  # JSON array
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        # JSON Lines: one object per line
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="forget10",
        help="full / forget01 / forget05 / forget10 / retain90 / retain95 / "
        "retain99 / world_facts / real_authors",
    )
    ap.add_argument(
        "--mode",
        choices=["answer", "qa", "chunk"],
        default="qa",
        help="answer: answer text only | qa: question+answer | "
        "chunk: group consecutive entries into one passage (per-author for 'full')",
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=20,
        help="only used with --mode chunk; in the 'full' config 20 QAs == 1 author",
    )
    ap.add_argument("--out", default=None, help="output .txt path")
    args = ap.parse_args()

    ds = load_tofu(args.config)

    lines = []
    if args.mode == "chunk":
        buf = []
        for i, ex in enumerate(ds):
            buf.append(f"{clean(ex['question'])} {clean(ex['answer'])}")
            if (i + 1) % args.chunk_size == 0:
                lines.append(" ".join(buf))
                buf = []
        if buf:
            lines.append(" ".join(buf))
    else:
        for ex in ds:
            if args.mode == "answer":
                lines.append(clean(ex["answer"]))
            else:  # qa
                lines.append(f"{clean(ex['question'])} {clean(ex['answer'])}")

    out = args.out or f"./datasets/tofu_{args.config}_{args.mode}.txt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(lines)} passages -> {out}")


if __name__ == "__main__":
    main()