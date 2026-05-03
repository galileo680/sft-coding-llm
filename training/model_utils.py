import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_for_training(model_name: str, load_in_4bit: bool, lora_cfg: dict) -> tuple:
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    tokenizer = load_tokenizer(model_name)

    return model, tokenizer


def load_model_for_inference(model_name: str, adapter_path: str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = load_tokenizer(model_name)

    return model, tokenizer