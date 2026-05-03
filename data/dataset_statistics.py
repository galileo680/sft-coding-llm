import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def detect_docstring_style(docstring: str) -> str:
    if ":param " in docstring or ":type " in docstring or ":returns:" in docstring:
        return "reST"
    if "Args:" in docstring or "Returns:" in docstring or "Raises:" in docstring:
        return "google"
    if "Parameters\n" in docstring or "Returns\n" in docstring:
        return "numpy"
    return "freeform"


def compute_stats(input_path: str, output_dir: str) -> None:
    records = load_jsonl(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    code_lengths = []
    doc_lengths_words = []
    doc_lengths_lines = []
    styles = Counter()
    func_name_prefixes = Counter()

    for rec in records:
        user_msg = rec["messages"][1]["content"]
        code_start = user_msg.find("```python\n") + len("```python\n")
        code_end = user_msg.rfind("\n```")
        code = user_msg[code_start:code_end]

        docstring = rec["messages"][2]["content"]

        code_lines = [l for l in code.strip().splitlines() if l.strip()]
        code_lengths.append(len(code_lines))

        doc_words = docstring.split()
        doc_lengths_words.append(len(doc_words))
        doc_lengths_lines.append(len(docstring.strip().splitlines()))

        styles[detect_docstring_style(docstring)] += 1

        name = rec.get("func_name", "")
        prefix = name.split(".")[-1].split("_")[0] if name else "unknown"
        func_name_prefixes[prefix] += 1

    code_arr = np.array(code_lengths)
    doc_arr = np.array(doc_lengths_words)

    stats = {
        "total_samples": len(records),
        "code_lines": {
            "mean": float(np.mean(code_arr)),
            "median": float(np.median(code_arr)),
            "std": float(np.std(code_arr)),
            "min": int(np.min(code_arr)),
            "max": int(np.max(code_arr)),
            "p25": float(np.percentile(code_arr, 25)),
            "p75": float(np.percentile(code_arr, 75)),
            "p95": float(np.percentile(code_arr, 95)),
        },
        "docstring_words": {
            "mean": float(np.mean(doc_arr)),
            "median": float(np.median(doc_arr)),
            "std": float(np.std(doc_arr)),
            "min": int(np.min(doc_arr)),
            "max": int(np.max(doc_arr)),
            "p25": float(np.percentile(doc_arr, 25)),
            "p75": float(np.percentile(doc_arr, 75)),
            "p95": float(np.percentile(doc_arr, 95)),
        },
        "docstring_styles": dict(styles.most_common()),
        "top_func_prefixes": dict(func_name_prefixes.most_common(20)),
    }

    with open(out / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    np.save(str(out / "code_lengths.npy"), code_arr)
    np.save(str(out / "doc_lengths.npy"), doc_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/formatted.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/processed/stats")
    args = parser.parse_args()
    compute_stats(args.input, args.output_dir)
