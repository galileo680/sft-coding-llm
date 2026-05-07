# SFT Fine-Tuning of Qwen2.5-Coder-1.5B for Docstring Generation

This project fine-tunes [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) using Supervised Fine-Tuning (SFT) with LoRA to specialize it in generating Python docstrings. The goal is to take a general-purpose coding assistant and improve its ability to produce clear, comprehensive docstrings for Python functions.

The project covers the full post-training pipeline: data curation from CodeSearchNet, quality filtering, SFT with QLoRA, hyperparameter ablation across LoRA ranks (16, 32, 64), and evaluation using BLEU, ROUGE-L, and qualitative analysis.

Fine Tuned model is available on [HuggingFace](https://huggingface.co/galileo680/qwen2.5-coder-1.5b-docstring-sft)

## Results

Fine-tuning consistently improved docstring generation quality across all configurations. The best model (LoRA rank 64) achieved a **2× improvement in ROUGE-L** and **6× improvement in BLEU** over the baseline.

| Experiment | LoRA Rank | BLEU | ROUGE-L | Eval Loss |
|---|---|---|---|---|
| Baseline (no fine-tuning) | — | 0.0063 | 0.1064 | — |
| SFT LoRA r=16  | 16 | 0.0318 | 0.2111 | 1.1104 |
| SFT LoRA r=32 | 32 | 0.0371 | 0.2136 | 1.1004 |
| SFT LoRA r=64 | 64 | 0.0381 | 0.2188 | 1.0910 |

**Key findings:**
- All fine-tuned models outperform the baseline, confirming that SFT effectively specializes the model for docstring generation.
- Increasing LoRA rank from 16 to 64 yields consistent but decreasing improvements: the jump from baseline to r=16 is much larger than from r=16 to r=64.
- Eval loss decreases monotonically with higher rank, suggesting the model benefits from additional adapter capacity on this task.
- Zero exact match rate across all models indicates the model generates paraphrased docstrings rather than memorizing training data.

## Dataset

The training data comes from [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) (Python subset), processed through a multi-stage filtering pipeline.

**Filtering pipeline**:

| Filter | Rejected |
|---|---|
| Short docstring (<10 words) | 116,133 |
| Code too long (>80 lines) | 8,976 |
| Syntax errors | 2,744 |
| Failed docstring removal | 3,321 |
| Code too short (<4 lines) | 869 |
| Trivial body (pass/...) | 76 |
| Trivial docstring | 6 |
| **Total rejected** | **132,125** |
| **Accepted** | **280,053** |

From the 280k accepted examples, 25,000 were sampled for training (22,500 train / 1,250 val / 1,250 test).


Each training example follows the Qwen2.5-Coder-Instruct chat template with a system prompt, the function code (with docstring removed) as the user message, and the original docstring as the assistant response.

## Training Setup

- **Model:** Qwen2.5-Coder-1.5B-Instruct
- **Method:** QLoRA (4-bit NF4 quantization + LoRA adapters)
- **LoRA targets:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training:** 3 epochs, batch size 4, gradient accumulation 4 (effective batch 16), cosine LR schedule
- **Hardware:** Google Colab A100 GPU
- **Frameworks:** Hugging Face TRL, PEFT, Transformers, W&B

**Training details per experiment:**

| Experiment | LoRA Rank | Alpha | LR | Train Loss | Eval Loss | Runtime |
|---|---|---|---|---|---|---|
| SFT Base | 16 | 32 | 2e-5 | 1.1295 | 1.1104 | 3h 19min |
| SFT r=32 | 32 | 64 | 2e-5 | 1.1109 | 1.1004 | 3h 30min |
| SFT r=64 | 64 | 128 | 2e-5 | 1.0885 | 1.0910 | 3h 23min |



## Reproducing Results

**1. Setup:**
```bash
git clone https://github.com/galileo680/sft-coding-llm.git
cd sft-coding-llm
pip install -r requirements.txt
```

**2. Prepare data:**
```bash
bash data/run_pipeline.sh
```

**3. Train:**
```bash
python -m training.train_sft --config configs/sft_base.yaml
```

**4. Evaluate:**
```bash
python -m evaluation.generate_samples \
    --model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --adapter-path outputs/sft_base/final \
    --test-path data/processed/splits/test.jsonl \
    --output-path results/sft_base/task_eval/generations.jsonl

python -m evaluation.run_task_eval \
    --input results/sft_base/task_eval/generations.jsonl \
    --output-dir results/sft_base/task_eval
```

## Tools

**Libraries:** [TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft), [Transformers](https://github.com/huggingface/transformers), [W&B](https://wandb.ai/)


## License

MIT
