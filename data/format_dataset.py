import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful coding assistant specialized in generating Python docstrings. "
    "Given a Python function, generate a clear and comprehensive docstring that describes "
    "what the function does, its parameters, and its return value."
)

USER_TEMPLATE = "Generate a Python docstring for the following function:\n\n```python\n{code}\n```"


def format_as_chat(func_without_docstring: str, docstring: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(code=func_without_docstring.strip())},
        {"role": "assistant", "content": docstring.strip()},
    ]


def format_dataset(input_path: str, output_path: str) -> int:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            messages = format_as_chat(row["func_without_docstring"], row["docstring"])
            record = {
                "messages": messages,
                "func_name": row["func_name"],
                "repository_name": row["repository_name"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/filtered.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/formatted.jsonl")
    args = parser.parse_args()
    format_dataset(args.input, args.output)
