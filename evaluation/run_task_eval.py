import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.metrics import compute_metrics


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(input_path: str, output_dir: str) -> dict:
    records = load_jsonl(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for rec in records:
        m = compute_metrics(rec["expected"], rec["generated"])
        m["func_name"] = rec.get("func_name", "")
        m["index"] = rec.get("index", -1)
        all_metrics.append(m)

    bleu_scores = [m["bleu"] for m in all_metrics]
    rouge_scores = [m["rouge_l"] for m in all_metrics]
    exact_matches = [m["exact_match"] for m in all_metrics]

    summary = {
        "num_samples": len(all_metrics),
        "bleu": {
            "mean": float(np.mean(bleu_scores)),
            "median": float(np.median(bleu_scores)),
            "std": float(np.std(bleu_scores)),
            "min": float(np.min(bleu_scores)),
            "max": float(np.max(bleu_scores)),
        },
        "rouge_l": {
            "mean": float(np.mean(rouge_scores)),
            "median": float(np.median(rouge_scores)),
            "std": float(np.std(rouge_scores)),
            "min": float(np.min(rouge_scores)),
            "max": float(np.max(rouge_scores)),
        },
        "exact_match_rate": float(np.mean(exact_matches)),
    }

    with open(out / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / "eval_per_sample.jsonl", "w") as f:
        for m in all_metrics:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    sorted_by_bleu = sorted(all_metrics, key=lambda x: x["bleu"])
    worst = sorted_by_bleu[:10]
    best = sorted_by_bleu[-10:]

    with open(out / "best_samples.json", "w") as f:
        json.dump(best, f, indent=2)

    with open(out / "worst_samples.json", "w") as f:
        json.dump(worst, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.input, args.output_dir)
