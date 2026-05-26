"""Quick sanity check: load the fine-tuned LoRA adapter and generate floor plans.

Run after training finishes:
    python quick_test.py

Outputs (in outputs/):
    - sample_<area>m2.json        — raw model output
    - sample_<area>m2_clean.json  — post-processed (drop these in your viewer!)
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from post_process import post_process

os.makedirs("outputs", exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_PATH = "./qwen-kalkulio-lora-v2/final"  # current best model

SYSTEM = "You are an expert architectural AI. Generate a valid JSON floor plan for a single-family house."

print("Loading tokenizer + base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()

# Try a few different prompts so we see if it generalizes at all
TEST_AREAS = [90.0, 120.0, 180.0]

for area in TEST_AREAS:
    print(f"\n{'=' * 60}")
    print(f"Prompt: area = {area} m²")
    print("=" * 60)

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Generate a floor plan for a house with an approximate area of {area}m2.",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8192,  # Qwen 152k vocab + dense JSON needs lots of room
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )

    # Always save the RAW model output so you can compare against the cleaned one
    raw_path = f"outputs/sample_{int(area)}m2.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"Raw output saved to: {raw_path}  ({len(raw)} chars)")

    # Parse + score + post-process
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"INVALID JSON: {e}")
        print(f"   first 600 chars of raw: {raw[:600]}")
        continue

    raw_walls = len(plan.get("steny", []))
    raw_openings = len(plan.get("otvory", []))
    raw_rooms = len(plan.get("prostory", []))
    raw_area = round(sum(p.get("plocha_m2", 0) for p in plan.get("prostory", []) if not p.get("venkovni", False)), 1)
    print(f"RAW    : walls={raw_walls} openings={raw_openings} rooms={raw_rooms} area={raw_area}m²")

    cleaned = post_process(plan, target_area=area)
    cleaned_walls = len(cleaned.get("steny", []))
    cleaned_openings = len(cleaned.get("otvory", []))
    cleaned_rooms = len(cleaned.get("prostory", []))
    cleaned_area = round(sum(p.get("plocha_m2", 0) for p in cleaned.get("prostory", [])), 1)
    print(f"CLEANED: walls={cleaned_walls} openings={cleaned_openings} rooms={cleaned_rooms} area={cleaned_area}m² (target {area})")

    clean_path = f"outputs/sample_{int(area)}m2_clean.json"
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, separators=(",", ":"), ensure_ascii=False)
    print(f"Cleaned output saved to: {clean_path}  (drop this in the viewer)")

print(f"\n{'=' * 60}")
print("Done. If 3/3 parsed, you're in great shape for tomorrow.")
print("=" * 60)
