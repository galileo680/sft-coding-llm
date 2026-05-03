import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compare(result_dirs: list[str], output_path: str) -> None:
    rows = []

    for rdir in result_dirs:
        rdir = Path(rdir)
        name = rdir.name

        row = {"experiment": name}

        task_eval = rdir / "task_eval" / "eval_summary.json"
        if task_eval.exists():
            summary = load_json(str(task_eval))
            row["bleu_mean"] = round(summary["bleu"]["mean"], 4)
            row["rouge_l_mean"] = round(summary["rouge_l"]["mean"], 4)
            row["exact_match"] = round(summary["exact_match_rate"], 4)

        humaneval = rdir / "humaneval" / "humaneval_results.json"
        if humaneval.exists():
            he_results = load_json(str(humaneval))
            if "humaneval" in he_results:
                row["humaneval_pass1"] = round(he_results["humaneval"]["pass@1"], 4)

        rows.append(row)

    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())
    cols = ["experiment"] + sorted(c for c in all_cols if c != "experiment")

    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, separator]

    for r in rows:
        vals = [str(r.get(c, "-")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    table = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Experiment Comparison\n\n")
        f.write(table + "\n")

    comparison = {"columns": cols, "rows": rows}
    json_path = str(Path(output_path).with_suffix(".json"))
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="results/tables/main_results.md")
    args = parser.parse_args()
    compare(args.result_dirs, args.output)
