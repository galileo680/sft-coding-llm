import argparse
import json
from pathlib import Path
from datasets import load_dataset


def download(output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("code_search_net", "python")

    for split_name in ["train", "validation", "test"]:
        split = ds[split_name]
        records = []
        for row in split:
            records.append({
                "func_name": row["func_name"],
                "whole_func_string": row["whole_func_string"],
                "func_code_string": row["func_code_string"],
                "func_documentation_string": row["func_documentation_string"],
                "repository_name": row["repository_name"],
                "func_path_in_repository": row["func_path_in_repository"],
            })

        out_file = output_path / f"{split_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()
    download(args.output_dir)