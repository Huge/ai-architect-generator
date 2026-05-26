import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Free speedups on Blackwell — TF32 matmul/cudnn is off by default in PyTorch 2.x
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
OUTPUT_DIR = "./qwen-kalkulio-lora-v3"
DATA_DIR = "data"

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Load Dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files={
    "train": os.path.join(DATA_DIR, "train.jsonl"),
    "validation": os.path.join(DATA_DIR, "valid.jsonl")
})

# 4. Load Model
print(f"Loading model {MODEL_ID} on GPU...")
# Using bfloat16 for RTX 6000 Blackwell
# We try to use flash_attention_2 if available, otherwise it falls back
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    print("✅ Using Flash Attention 2")
except (ValueError, ImportError):
    print("⚠️ Flash Attention 2 not available, falling back to SDPA (still fast!).")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )

# 5. LoRA Configuration
# Target all linear layers for perfect fine-tuning
lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# Note: do NOT wrap with get_peft_model here — SFTTrainer applies peft_config
# itself. Passing both a PeftModel AND peft_config raises a ValueError.

# 6. Training Arguments
# Tailored for RTX PRO 6000 (48GB VRAM)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    # v3 config: tuned for the larger dataset (~2500 examples after augmentation).
    # batch 4 × grad_accum 4 = effective 16. Qwen's 152k vocab caps per-device batch.
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,              # ~3% warmup — scales with total steps
    num_train_epochs=4,             # more data per epoch → fewer epochs needed
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    tf32=True,
    report_to="none",
    optim="adamw_torch_fused",
    max_length=4096,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

# 7. Initialize SFTTrainer
# TRL's SFTTrainer will automatically apply the tokenizer's chat template 
# since the data has a 'messages' column prepared by prepare_data.py
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    args=training_args,
)

# 8. Train!
print("Starting training...")
trainer.train()

# 9. Save Final Model
print("Saving final model adapter...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

print("✅ Training complete!")
