import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Optional


def count_code_lines(code: str) -> int:
    lines = [l.strip() for l in code.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
    return len(lines)


def is_trivial_body(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = node.body
            if len(body) == 1:
                stmt = body[0]
                if isinstance(stmt, ast.Pass):
                    return True
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str)):
                    return True
                if isinstance(stmt, ast.Return) and stmt.value is None:
                    return True
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                    return True
            break
    return False


def is_docstring_trivial(func_name: str, docstring: str) -> bool:
    normalized_name = func_name.split(".")[-1].replace("_", " ").lower().strip()
    normalized_doc = docstring.lower().strip().rstrip(".")
    if normalized_doc == normalized_name:
        return True
    if normalized_doc in ("todo", "fixme", "hack", "xxx", "pass", "stub", "placeholder"):
        return True
    if re.match(r"^(get|set|init|create|make|do|run|execute)\s+" + re.escape(normalized_name.split()[-1]) + r"\.?$", normalized_doc):
        return True
    return False


def has_syntax_error(code: str) -> bool:
    try:
        ast.parse(code)
        return False
    except SyntaxError:
        return True


def remove_docstring_from_func(whole_func: str) -> Optional[str]:
    try:
        tree = ast.parse(whole_func)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if (node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant, ast.Str))):

                doc_node = node.body[0]
                lines = whole_func.splitlines(keepends=True)
                start_line = doc_node.lineno - 1
                end_line = doc_node.end_lineno

                before = lines[:start_line]
                after = lines[end_line:]

                result = "".join(before) + "".join(after)

                remaining_lines = [l for l in result.strip().splitlines() if l.strip()]
                if len(remaining_lines) < 2:
                    return None

                return result
            break
    return None


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def filter_dataset(input_path: str, output_path: str, min_doc_words: int, min_code_lines: int, max_code_lines: int) -> dict:
    stats = {
        "total": 0,
        "rejected_short_docstring": 0,
        "rejected_short_code": 0,
        "rejected_long_code": 0,
        "rejected_trivial_docstring": 0,
        "rejected_trivial_body": 0,
        "rejected_syntax_error": 0,
        "rejected_no_docstring_removal": 0,
        "rejected_duplicate": 0,
        "accepted": 0,
    }

    seen_hashes = set()
    accepted = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stats["total"] += 1

            docstring = row["func_documentation_string"].strip()
            whole_func = row["whole_func_string"]
            func_name = row["func_name"]

            if len(docstring.split()) < min_doc_words:
                stats["rejected_short_docstring"] += 1
                continue

            if has_syntax_error(whole_func):
                stats["rejected_syntax_error"] += 1
                continue

            code_lines = count_code_lines(whole_func)
            if code_lines < min_code_lines:
                stats["rejected_short_code"] += 1
                continue
            if code_lines > max_code_lines:
                stats["rejected_long_code"] += 1
                continue

            if is_docstring_trivial(func_name, docstring):
                stats["rejected_trivial_docstring"] += 1
                continue

            if is_trivial_body(whole_func):
                stats["rejected_trivial_body"] += 1
                continue

            func_without_doc = remove_docstring_from_func(whole_func)
            if func_without_doc is None:
                stats["rejected_no_docstring_removal"] += 1
                continue

            code_hash = compute_hash(func_without_doc.strip())
            if code_hash in seen_hashes:
                stats["rejected_duplicate"] += 1
                continue
            seen_hashes.add(code_hash)

            accepted.append({
                "func_name": func_name,
                "whole_func_string": whole_func,
                "func_without_docstring": func_without_doc,
                "docstring": docstring,
                "repository_name": row["repository_name"],
                "func_path_in_repository": row["func_path_in_repository"],
            })

    stats["accepted"] = len(accepted)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/train.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/filtered.jsonl")
    parser.add_argument("--min-doc-words", type=int, default=10)
    parser.add_argument("--min-code-lines", type=int, default=4)
    parser.add_argument("--max-code-lines", type=int, default=80)
    args = parser.parse_args()

    stats = filter_dataset(args.input, args.output, args.min_doc_words, args.min_code_lines, args.max_code_lines)

    stats_path = Path(args.output).parent / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
