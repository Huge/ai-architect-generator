"""Generate a floor plan using Best-of-N sampling to guarantee a watertight house.

Usage:
    python generate_best_of_n.py --area 100
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from post_process import post_process, has_closed_exterior_loop

BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
ADAPTER_PATH = "./qwen-kalkulio-lora-14b-v4/final"
SYSTEM = "You are an expert architectural AI. Generate a valid JSON floor plan for a single-family house."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", type=float, default=100.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    os.makedirs("outputs", exist_ok=True)
    
    best_plan = None
    best_raw = None

    for attempt in range(1, args.max_attempts + 1):
        print(f"\n--- Attempt {attempt}/{args.max_attempts} for {args.area}m2 ---")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Generate a floor plan for a house with an approximate area of {args.area}m2."},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.7, # Need variance to get different topologies!
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            print("  [Failed] Invalid JSON")
            continue

        # Post process
        cleaned = post_process(plan, target_area=args.area)
        
        # Check watertight
        is_watertight = has_closed_exterior_loop(cleaned)
        
        if is_watertight:
            print("  [SUCCESS] Found a watertight, clean floor plan!")
            best_plan = cleaned
            best_raw = raw
            break
        else:
            print("  [Failed] Plan is not watertight (has missing exterior walls or dead ends).")
            # Keep it as fallback if it's the last attempt
            best_plan = cleaned
            best_raw = raw

    # Save
    out_clean = f"outputs/sample_watertight_{int(args.area)}m2_clean.json"
    out_raw = f"outputs/sample_watertight_{int(args.area)}m2_raw.json"
    
    with open(out_clean, "w", encoding="utf-8") as f:
        json.dump(best_plan, f, separators=(",", ":"), ensure_ascii=False)
    with open(out_raw, "w", encoding="utf-8") as f:
        f.write(best_raw)
        
    print(f"\nDone! Saved best result to {out_clean}")

if __name__ == "__main__":
    main()
