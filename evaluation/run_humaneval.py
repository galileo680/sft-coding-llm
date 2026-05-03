import argparse
import subprocess
import json
from pathlib import Path


def run_humaneval(
    model_name: str,
    adapter_path: str,
    output_dir: str,
    n_samples: int,
    batch_size: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if adapter_path:
        from training.model_utils import load_model_for_inference
        from peft import PeftModel
        import torch

        merged_path = str(out / "merged_model")
        if not Path(merged_path).exists():
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)

            del model
            torch.cuda.empty_cache()

        eval_model = merged_path
    else:
        eval_model = model_name

    cmd = [
        "accelerate", "launch",
        "--main_process_port", "29501",
        "-m", "bigcode_eval",
        "--model", eval_model,
        "--tasks", "humaneval",
        "--n_samples", str(n_samples),
        "--batch_size", str(batch_size),
        "--allow_code_execution",
        "--trust_remote_code",
        "--save_generations",
        "--save_generations_path", str(out / "humaneval_generations.json"),
        "--metric_output_path", str(out / "humaneval_results.json"),
        "--max_length_generation", "512",
        "--precision", "bf16",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    run_humaneval(args.model_name, args.adapter_path, args.output_dir, args.n_samples, args.batch_size)
