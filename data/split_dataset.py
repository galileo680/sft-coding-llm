import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_dataset(
    input_path: str,
    output_dir: str,
    max_samples: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    records = load_jsonl(input_path)
    random.seed(seed)
    random.shuffle(records)

    if max_samples > 0 and len(records) > max_samples:
        records = records[:max_samples]

    total = len(records)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    n_train = total - n_val - n_test

    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    out = Path(output_dir)
    save_jsonl(train, str(out / "train.jsonl"))
    save_jsonl(val, str(out / "val.jsonl"))
    save_jsonl(test, str(out / "test.jsonl"))

    stats = {"total": total, "train": len(train), "val": len(val), "test": len(test)}

    with open(out / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/formatted.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/processed/splits")
    parser.add_argument("--max-samples", type=int, default=25000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.input, args.output_dir, args.max_samples, args.val_ratio, args.test_ratio, args.seed)
