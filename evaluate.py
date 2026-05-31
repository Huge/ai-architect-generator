"""Evaluation harness for the fine-tuned Kalkulio model.

For each requested area (60..220 m² across N samples), generate a floor plan,
score it raw, then run post_process.py and score it again. Aggregates into a
report you can use to compare adapters (v1 vs v2 vs future runs).

Usage:
    python evaluate.py                                  # default settings
    python evaluate.py --num-samples 20                 # more samples
    python evaluate.py --adapter qwen-kalkulio-lora/final   # different model
    python evaluate.py --report eval_v2.json            # custom report path
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from post_process import post_process, _polygon_area


BASE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEFAULT_ADAPTER = "./qwen-kalkulio-lora-14b-v4/final"

SYSTEM = "You are an expert architectural AI. Generate a valid JSON floor plan for a single-family house."
USER_TEMPLATE = "Generate a floor plan for a house with an approximate area of {area}m2."

REQUIRED_KEYS = ("steny", "otvory", "prostory")


# ----- Scoring -----------------------------------------------------------------

def score_plan(plan: dict, target_area: float) -> dict:
    """Score a parsed plan against a target area.

    Returns a dict of metrics. NaN-safe — missing fields are treated as zeros.
    """
    walls = plan.get("steny") or []
    openings = plan.get("otvory") or []
    rooms = plan.get("prostory") or []

    has_keys = all(k in plan for k in REQUIRED_KEYS)

    # Wall connectivity: how many wall endpoints are shared with another endpoint?
    eps = 0.15
    points = []
    for w in walls:
        if "od" in w and "do" in w:
            points.extend([tuple(w["od"]), tuple(w["do"])])
    shared = 0
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i == j:
                continue
            if abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) < eps:
                shared += 1
                break
    wall_connectivity = (shared / len(points)) if points else 0.0

    # Polygons closed (first vertex == last vertex)
    closed = 0
    polygon_count = 0
    for r in rooms:
        poly = r.get("polygon") or []
        if len(poly) >= 3:
            polygon_count += 1
            if poly[0] == poly[-1]:
                closed += 1
    polygon_closure = (closed / polygon_count) if polygon_count else 0.0

    # Openings on valid walls (host wall exists)
    wall_ids = {w["id"] for w in walls if "id" in w}
    valid_openings = sum(1 for o in openings if o.get("stena") in wall_ids)
    opening_validity = (valid_openings / len(openings)) if openings else 1.0

    # Orphan rooms (any polygon vertex far outside wall envelope)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if xs and ys:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        orphan = 0
        for r in rooms:
            poly = r.get("polygon") or []
            if not poly:
                continue
            outside = sum(
                1 for p in poly
                if p[0] < min_x - 0.15 or p[0] > max_x + 0.15
                or p[1] < min_y - 0.15 or p[1] > max_y + 0.15
            )
            if outside / len(poly) > 0.25:
                orphan += 1
    else:
        orphan = 0
    orphan_ratio = (orphan / len(rooms)) if rooms else 0.0

    # Area: prefer recomputed-from-polygon to catch any stale plocha_m2
    interior_area = sum(
        _polygon_area(r["polygon"]) for r in rooms
        if not r.get("venkovni", False) and r.get("polygon")
    )
    area_abs_error = abs(interior_area - target_area)
    area_rel_error = (area_abs_error / target_area) if target_area > 0 else 0.0

    return {
        "has_keys": has_keys,
        "num_walls": len(walls),
        "num_openings": len(openings),
        "num_rooms": len(rooms),
        "wall_connectivity": round(wall_connectivity, 3),
        "polygon_closure": round(polygon_closure, 3),
        "opening_validity": round(opening_validity, 3),
        "orphan_room_ratio": round(orphan_ratio, 3),
        "interior_area_m2": round(interior_area, 2),
        "area_abs_error_m2": round(area_abs_error, 2),
        "area_rel_error": round(area_rel_error, 3),
    }


# ----- Inference ---------------------------------------------------------------

def load_model(adapter_path: str):
    print(f"Loading tokenizer + base model {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print(f"Loading adapter {adapter_path}...")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, area: float, max_new_tokens: int = 8192,
                 temperature: float = 0.7, top_p: float = 0.9) -> Optional[dict]:
    """Generate one plan. Returns parsed dict, or None if invalid JSON."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(area=area)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


# ----- Main --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Path to the LoRA adapter dir")
    p.add_argument("--num-samples", type=int, default=10, help="Number of plans to generate")
    p.add_argument("--areas", type=str, default="70,90,110,130,150,170,190",
                   help="Comma-separated areas to cycle through (m²)")
    p.add_argument("--report", default="eval_report.json", help="Path to write JSON report")
    p.add_argument("--save-samples", default="eval_outputs",
                   help="Directory for raw + cleaned sample files (empty string to skip)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    areas = [float(a) for a in args.areas.split(",")]

    if args.save_samples:
        os.makedirs(args.save_samples, exist_ok=True)

    model, tokenizer = load_model(args.adapter)

    rows: list[dict] = []
    invalid_count = 0
    t_total_start = time.time()

    for i in range(args.num_samples):
        target = areas[i % len(areas)]
        t0 = time.time()
        plan, raw = generate_one(model, tokenizer, target)
        gen_time = time.time() - t0

        if plan is None:
            invalid_count += 1
            print(f"[{i+1}/{args.num_samples}] target={target:.0f} m² | INVALID JSON ({len(raw)} chars, {gen_time:.1f}s)")
            rows.append({"i": i, "target_area_m2": target, "valid_json": False, "gen_time_s": round(gen_time, 1)})
            if args.save_samples:
                with open(os.path.join(args.save_samples, f"raw_{i:03d}_{int(target)}m2.txt"), "w") as f:
                    f.write(raw)
            continue

        raw_scores = score_plan(plan, target)
        cleaned = post_process(json.loads(json.dumps(plan)), target_area=target)  # deepcopy via json
        clean_scores = score_plan(cleaned, target)

        print(
            f"[{i+1}/{args.num_samples}] target={target:.0f} m² | "
            f"raw: area={raw_scores['interior_area_m2']:.0f} rooms={raw_scores['num_rooms']} orphans={raw_scores['orphan_room_ratio']:.2f} | "
            f"clean: area={clean_scores['interior_area_m2']:.0f} rooms={clean_scores['num_rooms']} orphans={clean_scores['orphan_room_ratio']:.2f} | "
            f"{gen_time:.1f}s"
        )

        row = {"i": i, "target_area_m2": target, "valid_json": True, "gen_time_s": round(gen_time, 1)}
        row.update({f"raw_{k}": v for k, v in raw_scores.items()})
        row.update({f"clean_{k}": v for k, v in clean_scores.items()})
        rows.append(row)

        if args.save_samples:
            with open(os.path.join(args.save_samples, f"raw_{i:03d}_{int(target)}m2.json"), "w") as f:
                json.dump(plan, f, separators=(",", ":"), ensure_ascii=False)
            with open(os.path.join(args.save_samples, f"clean_{i:03d}_{int(target)}m2.json"), "w") as f:
                json.dump(cleaned, f, separators=(",", ":"), ensure_ascii=False)

    total_time = time.time() - t_total_start

    # Aggregates
    valid_rows = [r for r in rows if r.get("valid_json")]
    n = len(rows)
    n_valid = len(valid_rows)

    def avg(key: str) -> float:
        vals = [r[key] for r in valid_rows if key in r]
        return round(statistics.mean(vals), 3) if vals else 0.0

    summary = {
        "adapter": args.adapter,
        "num_samples": n,
        "valid_json_rate": round(n_valid / n, 3) if n else 0.0,
        "total_time_s": round(total_time, 1),
        "avg_gen_time_s": round(statistics.mean(r["gen_time_s"] for r in rows), 2) if rows else 0.0,

        # Raw model output
        "raw_avg_rooms": avg("raw_num_rooms"),
        "raw_avg_walls": avg("raw_num_walls"),
        "raw_avg_area_rel_error": avg("raw_area_rel_error"),
        "raw_avg_orphan_ratio": avg("raw_orphan_room_ratio"),
        "raw_avg_opening_validity": avg("raw_opening_validity"),
        "raw_avg_polygon_closure": avg("raw_polygon_closure"),
        "raw_avg_wall_connectivity": avg("raw_wall_connectivity"),

        # After post-processor
        "clean_avg_rooms": avg("clean_num_rooms"),
        "clean_avg_walls": avg("clean_num_walls"),
        "clean_avg_area_rel_error": avg("clean_area_rel_error"),
        "clean_avg_orphan_ratio": avg("clean_orphan_room_ratio"),
        "clean_avg_opening_validity": avg("clean_opening_validity"),
        "clean_avg_polygon_closure": avg("clean_polygon_closure"),
        "clean_avg_wall_connectivity": avg("clean_wall_connectivity"),
    }

    report = {"summary": summary, "rows": rows}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty print summary
    print("\n" + "=" * 60)
    print(f"SUMMARY  (adapter: {summary['adapter']})")
    print("=" * 60)
    print(f"  samples:              {summary['num_samples']}  (valid JSON: {summary['valid_json_rate']*100:.0f}%)")
    print(f"  total time:           {summary['total_time_s']}s  (avg {summary['avg_gen_time_s']}s / sample)")
    print()
    print(f"  {'metric':30s} {'raw':>10s} {'cleaned':>10s}  delta")
    print(f"  {'-'*30} {'-'*10} {'-'*10}  -----")
    for label, raw_k, clean_k in [
        ("area_rel_error (lower=better)", "raw_avg_area_rel_error", "clean_avg_area_rel_error"),
        ("orphan_room_ratio (lower)",     "raw_avg_orphan_ratio",   "clean_avg_orphan_ratio"),
        ("opening_validity (higher)",     "raw_avg_opening_validity","clean_avg_opening_validity"),
        ("polygon_closure (higher)",      "raw_avg_polygon_closure", "clean_avg_polygon_closure"),
        ("wall_connectivity (higher)",    "raw_avg_wall_connectivity","clean_avg_wall_connectivity"),
        ("num_rooms",                     "raw_avg_rooms",          "clean_avg_rooms"),
    ]:
        r, c = summary[raw_k], summary[clean_k]
        d = c - r
        arrow = "+" if d >= 0 else ""
        print(f"  {label:30s} {r:>10} {c:>10}  {arrow}{d:.3f}")
    print(f"\nReport written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
