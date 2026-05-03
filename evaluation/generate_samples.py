import argparse
import json
from pathlib import Path

import torch
from training.model_utils import load_model_for_inference, load_tokenizer


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_docstring(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


def run(
    model_name: str,
    adapter_path: str,
    test_path: str,
    output_path: str,
    max_samples: int,
    max_new_tokens: int,
) -> None:
    if adapter_path:
        model, tokenizer = load_model_for_inference(model_name, adapter_path)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        tokenizer = load_tokenizer(model_name)

    records = load_jsonl(test_path)
    if max_samples > 0:
        records = records[:max_samples]

    results = []
    for i, rec in enumerate(records):
        messages_input = rec["messages"][:2]
        expected = rec["messages"][2]["content"]

        generated = generate_docstring(model, tokenizer, messages_input, max_new_tokens)

        results.append({
            "func_name": rec.get("func_name", ""),
            "expected": expected,
            "generated": generated,
            "index": i,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--test-path", type=str, default="data/processed/splits/test.jsonl")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    run(args.model_name, args.adapter_path, args.test_path, args.output_path, args.max_samples, args.max_new_tokens)
