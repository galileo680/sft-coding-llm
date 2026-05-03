import json
import random
from transformers import TrainerCallback
import wandb


class GenerationCallback(TrainerCallback):

    def __init__(self, tokenizer, val_path: str, num_samples: int = 5, max_new_tokens: int = 256, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        random.seed(seed)

        with open(val_path, "r", encoding="utf-8") as f:
            all_records = [json.loads(line) for line in f]

        self.samples = random.sample(all_records, min(num_samples, len(all_records)))

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        model.eval()
        table_data = []

        for rec in self.samples:
            messages = rec["messages"][:2]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            expected = rec["messages"][2]["content"]
            func_name = rec.get("func_name", "unknown")

            table_data.append([func_name, expected[:500], generated[:500]])

        table = wandb.Table(
            columns=["function", "expected", "generated"],
            data=table_data,
        )
        wandb.log({"generation_samples": table, "global_step": state.global_step})

        model.train()
