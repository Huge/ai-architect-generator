"""Quick sanity check: load the fine-tuned LoRA adapter and generate ONE floor plan.

Run after training finishes:
    python quick_test.py

Outputs:
    - prints a short preview to the terminal
    - saves the FULL raw JSON of each generation to outputs/sample_<area>m2.json
      (compact, exactly as the model produced it — drop these into your viewer)
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.makedirs("outputs", exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_PATH = "./qwen-kalkulio-lora/final"

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
            max_new_tokens=4096,  # floor-plan JSON can be 4-6k chars
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )

    # Always save the RAW model output so you can load it in the viewer
    raw_path = f"outputs/sample_{int(area)}m2.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"Raw output saved to: {raw_path}  ({len(raw)} chars)")

    # Score it
    try:
        plan = json.loads(raw)
        keys = set(plan.keys())
        expected = {"steny", "otvory", "prostory"}
        missing = expected - keys
        extra = keys - expected
        n_walls = len(plan.get("steny", []))
        n_openings = len(plan.get("otvory", []))
        n_rooms = len(plan.get("prostory", []))
        total_area = round(sum(p.get("plocha_m2", 0) for p in plan.get("prostory", [])), 1)
        print(f"PARSED OK. walls={n_walls} openings={n_openings} rooms={n_rooms} total_area={total_area}m2")
        if missing:
            print(f"   missing keys: {missing}")
        if extra:
            print(f"   extra keys: {extra}")
    except json.JSONDecodeError as e:
        print(f"INVALID JSON: {e}")
        print(f"   first 600 chars of raw: {raw[:600]}")

print(f"\n{'=' * 60}")
print("Done. If 3/3 parsed, you're in great shape for tomorrow.")
print("=" * 60)
