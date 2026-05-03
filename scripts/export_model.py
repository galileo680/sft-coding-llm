import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def merge_and_save(model_name: str, adapter_path: str, output_path: str, push_to_hub: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        tokenizer.push_to_hub(push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--push-to-hub", type=str, default="")
    args = parser.parse_args()
    merge_and_save(args.model_name, args.adapter_path, args.output_path, args.push_to_hub)
