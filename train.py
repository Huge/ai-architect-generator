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

# Detect distributed (DDP) launch via torchrun. When WORLD_SIZE > 1, each
# process trains a full copy of the model on its own GPU (data parallelism).
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
IS_DDP = WORLD_SIZE > 1
IS_MAIN = LOCAL_RANK == 0

# 1. Configuration
MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
OUTPUT_DIR = "./qwen-kalkulio-lora-14b-v4"
DATA_DIR = "data"

if IS_MAIN:
    print(f"🖥️  WORLD_SIZE={WORLD_SIZE}  ({'DDP data-parallel' if IS_DDP else 'single-GPU'})")

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
# CRITICAL: under DDP each process must place the FULL model on its own GPU
# (device_map={"": local_rank}). device_map="auto" is for sharding ONE model
# across GPUs and conflicts with DDP — only use it for single-process runs.
device_map = {"": LOCAL_RANK} if IS_DDP else "auto"

# Using bfloat16 for RTX 6000 Blackwell
# We try to use flash_attention_2 if available, otherwise it falls back
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2"
    )
    if IS_MAIN:
        print("✅ Using Flash Attention 2")
except (ValueError, ImportError):
    if IS_MAIN:
        print("⚠️ Flash Attention 2 not available, falling back to SDPA (still fast!).")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device_map,
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
# 14B v4 config for RTX PRO 6000 (96GB). 14B weights bf16 ≈ 28GB; the heavy
# term is the lm_head logits over 152k vocab. batch=2 + gradient checkpointing
# keeps peak VRAM ~60-70GB per GPU.
#
# Effective batch = per_device_batch × grad_accum × WORLD_SIZE. We auto-scale
# grad_accum down as we add GPUs so the effective batch stays ~16-18 regardless
# of GPU count (otherwise 3 GPUs would push it to 48 and starve us of steps).
PER_DEVICE_BATCH = 2
TARGET_EFFECTIVE_BATCH = 16
grad_accum = max(1, round(TARGET_EFFECTIVE_BATCH / (PER_DEVICE_BATCH * WORLD_SIZE)))
effective_batch = PER_DEVICE_BATCH * grad_accum * WORLD_SIZE
if IS_MAIN:
    print(f"📦 per_device={PER_DEVICE_BATCH} × grad_accum={grad_accum} × world={WORLD_SIZE} "
          f"= effective batch {effective_batch}")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=grad_accum,
    gradient_checkpointing=True,
    # DDP + gradient checkpointing needs this or it errors on unused params
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    learning_rate=1e-4,             # lower LR for the larger model
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
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
    dataloader_num_workers=8,       # 32-core Threadripper — plenty of headroom
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
# Auto-resume: if a checkpoint already exists in OUTPUT_DIR (e.g. after a
# server crash/reboot), pick up where we left off instead of restarting.
from transformers.trainer_utils import get_last_checkpoint

last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint:
    if IS_MAIN:
        print(f"🔄 Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting training...")
    trainer.train()

# 9. Save Final Model
print("Saving final model adapter...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

print("✅ Training complete!")
