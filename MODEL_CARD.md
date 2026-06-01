---
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - trl
  - sft
  - floor-plan
  - architecture
  - json-generation
  - kalkulio
language:
  - en
  - cs
---

# Kalkulio AI Architect — Qwen2.5-Coder-14B LoRA

A LoRA adapter that fine-tunes **Qwen/Qwen2.5-Coder-14B-Instruct** to generate
valid JSON floor plans for single-family houses, built for the **Kalkulio AI
Challenge**.

Given a simple target-area prompt (e.g. *"a house of about 90 m²"*), the model
outputs a structured **Kalkulio-format JSON** floor plan — walls (`steny`),
openings (`otvory`), and rooms (`prostory`) with polygons, areas, and Czech
room labels.

> **Important:** this adapter is designed to be paired with the project's
> **deterministic geometric post-processor**, which rebuilds walls from room
> polygons to guarantee watertight geometry and exact area. The post-processor
> and full Gradio app live in the project repo:
> 👉 https://github.com/naitik0009/ai-architect-generator

## Model details

- **Developed by:** Naitik Kumar Rauniyar
- **Base model:** [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
- **Adapter:** LoRA (PEFT), rank 128, α 256, dropout 0.05, all linear projections
- **Task:** structured JSON floor-plan generation
- **Languages:** English prompts, Czech room labels
- **License:** Apache 2.0 (inherits the base model)
- **Repository:** https://github.com/naitik0009/ai-architect-generator

## Intended use

Generate single-family house floor plans (most reliable in the **60–130 m²**
range) as Kalkulio-format JSON. The raw output should be passed through the
project's `post_process.py` for guaranteed geometric validity.

### How to use

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-Coder-14B-Instruct"
ADAPTER = "naitik12kumar/qwen-kalkulio-lora-14b-v4"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()

messages = [
    {"role": "system", "content": "You are an expert architectural AI. Generate a valid JSON floor plan for a single-family house."},
    {"role": "user", "content": "Generate a floor plan for a house with an approximate area of 90m2."},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=3072, do_sample=True, temperature=0.3, top_p=0.9)
raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
plan = json.loads(raw)

# Recommended: clean with the project's post-processor for a watertight result
# from post_process import post_process
# plan = post_process(plan, target_area=90)
```

## Training

- **Data:** the 60 real Kalkulio houses + Gemini-generated synthetic plans +
  70 hand-authored plans, expanded ×8 via geometric augmentation
  (rotation × mirror).
- **Method:** LoRA SFT (TRL `SFTTrainer`), 2 epochs, learning rate 1e-4 with
  cosine schedule, effective batch ≈ 16–18, bf16, gradient checkpointing.
- **Hardware:** 3× NVIDIA RTX PRO 6000 Blackwell, data-parallel (DDP), ~1.5 h.

## Evaluation

Measured on **12 generated plans spanning 70–190 m²**, scoring the raw model
output vs. the same plans after the deterministic post-processor:

| Metric | Raw model | After post-process |
|--------|-----------|--------------------|
| Valid JSON | 100% | 100% |
| **Watertight rate** | 83% | **100%** |
| **Mean area error** | 27.7% | **0.0%** |
| Orphan rooms | 1.2% | **0%** |
| Polygon closure | 91% | **100%** |
| Wall connectivity | 99% | **100%** |

The model reliably emits valid, mostly-watertight JSON; the post-processor
closes the remaining gaps so **every** plan ends up watertight and exactly
area-matched.

## Limitations

- Raw outputs benefit from the post-processor for guaranteed validity; used
  standalone, a minority of plans are not watertight.
- Most reliable at **60–130 m²**. Larger plans (>150 m²) can show disconnected
  room clusters or oversized rooms.
- Room labels are in Czech (Kalkulio convention).

### Framework versions

- PEFT 0.19.1
- TRL, Transformers, PyTorch (CUDA)
