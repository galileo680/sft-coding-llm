import argparse
import json
import yaml
from pathlib import Path
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from training.model_utils import load_model_for_training

import wandb


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(path: str) -> Dataset:
    records = load_jsonl(path)
    return Dataset.from_list(records)


def train(config_path: str) -> None:
    cfg = load_config(config_path)

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )

    model, tokenizer = load_model_for_training(
        model_name=cfg["model"]["name"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        lora_cfg=cfg["lora"],
    )

    train_dataset = prepare_dataset(cfg["data"]["train_path"])
    val_dataset = prepare_dataset(cfg["data"]["val_path"])

    output_dir = cfg["output"]["dir"]
    tcfg = cfg["training"]

    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=cfg["model"]["max_seq_length"],
        num_train_epochs=tcfg["num_epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_steps=int(tcfg["warmup_ratio"] * 1000),
        weight_decay=tcfg["weight_decay"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        save_steps=tcfg["save_steps"],
        eval_steps=tcfg["eval_steps"],
        eval_strategy="steps",
        logging_steps=tcfg["logging_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        seed=tcfg["seed"],
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir + "/final")

    if cfg["output"].get("hub_model_id"):
        trainer.push_to_hub(cfg["output"]["hub_model_id"])

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)