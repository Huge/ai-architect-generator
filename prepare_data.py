"""Improved data prep for Kalkulio floor plans.

Compared to the original pipeline:
  - Adds rotation augmentation (0°, 90°, 180°, 270°)
  - Combines rotation × mirror × scale → ~24 variants per house
  - 8 prompt variants (English + Slovak, formal + casual)
  - Holds out whole houses for validation (no leakage from augmented twins)
  - Area-balanced oversampling: large houses (>120 m²) get 3x weight to fight
    the model's tendency to regress to the dataset median (~78 m²)

Usage:
    python prepare_data.py

Tunables at the top of the file.
"""

from __future__ import annotations

import copy
import json
import os
import random

INPUT_DIR = "raw_data"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Tunables ---------------------------------------------------------------

VALIDATION_HOUSES = 7            # Hold out N whole houses for validation
SCALES = [0.90, 1.00, 1.10]      # 3 scale variants
MIRRORS = [None, "x", "y"]       # 3 mirror modes
ROTATIONS = [0, 90, 180, 270]    # 4 rotations
# → 3 × 3 × 4 = 36 variants per house

# Area-based oversampling weights — fight the regression-to-mean bias
def area_weight(area_m2: float) -> int:
    if area_m2 >= 150:
        return 4
    if area_m2 >= 120:
        return 3
    if area_m2 >= 100:
        return 2
    return 1

SEED = 42

# ----- Prompt variants --------------------------------------------------------

SYSTEM_PROMPTS = [
    "You are an expert architectural AI. Generate a valid JSON floor plan for a single-family house.",
    "You are an architect specializing in single-family houses. Output a structured JSON floor plan.",
    "You design residential floor plans. Respond with compact JSON only — no commentary.",
]

USER_PROMPTS = [
    "Generate a floor plan for a house with an approximate area of {area}m2.",
    "Design a single-family house of about {area} square meters.",
    "I need a {area} m² family house, please generate a floor plan.",
    "Create a floor plan for a {area} square meter family home.",
    "{area}m² single-family house — generate the floor plan JSON.",
    "Show me a floor plan for a house with roughly {area} square meters.",
    "Návrh rodinného domu o ploše přibližně {area} m².",         # Slovak / Czech
    "Vytvor půdorys rodinného domu o ploše {area} m².",          # Slovak / Czech
]

# ----- Helpers ----------------------------------------------------------------

def round_floats(obj):
    if isinstance(obj, float):
        return round(obj, 2)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


def format_conversation(plan: dict, area_m2: float, rng: random.Random) -> dict:
    cleaned = {
        "steny": plan.get("steny", []),
        "otvory": plan.get("otvory", []),
        "prostory": plan.get("prostory", []),
    }
    cleaned = round_floats(cleaned)
    compact = json.dumps(cleaned, separators=(",", ":"), ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": rng.choice(SYSTEM_PROMPTS)},
            {"role": "user", "content": rng.choice(USER_PROMPTS).format(area=area_m2)},
            {"role": "assistant", "content": compact},
        ]
    }


# ----- Geometric augmentations ------------------------------------------------

def _apply_to_points(plan: dict, fn) -> dict:
    new = copy.deepcopy(plan)
    for w in new.get("steny", []):
        w["od"] = [round(c, 4) for c in fn(w["od"])]
        w["do"] = [round(c, 4) for c in fn(w["do"])]
    for r in new.get("prostory", []):
        r["polygon"] = [[round(c, 4) for c in fn(p)] for p in r["polygon"]]
    return new


def aug_scale(plan: dict, factor: float) -> dict:
    new = _apply_to_points(plan, lambda p: (p[0] * factor, p[1] * factor))
    for r in new.get("prostory", []):
        r["plocha_m2"] = round(r.get("plocha_m2", 0) * factor * factor, 2)
    return new


def aug_mirror(plan: dict, axis: str) -> dict:
    if axis == "x":
        return _apply_to_points(plan, lambda p: (-p[0], p[1]))
    if axis == "y":
        return _apply_to_points(plan, lambda p: (p[0], -p[1]))
    return copy.deepcopy(plan)


def aug_rotate(plan: dict, degrees: int) -> dict:
    if degrees == 0:
        return copy.deepcopy(plan)
    if degrees == 90:
        return _apply_to_points(plan, lambda p: (-p[1], p[0]))
    if degrees == 180:
        return _apply_to_points(plan, lambda p: (-p[0], -p[1]))
    if degrees == 270:
        return _apply_to_points(plan, lambda p: (p[1], -p[0]))
    raise ValueError(f"Unsupported rotation: {degrees}")


def all_variants(base: dict) -> list[tuple[dict, float]]:
    """Return (plan, area) for every (rotation, mirror, scale) combination."""
    out = []
    base_area = sum(p.get("plocha_m2", 0) for p in base.get("prostory", []))
    for rot in ROTATIONS:
        for mirror in MIRRORS:
            for scale in SCALES:
                plan = base
                plan = aug_rotate(plan, rot)
                plan = aug_mirror(plan, mirror) if mirror else plan
                plan = aug_scale(plan, scale)
                area = round(base_area * scale * scale, 1)
                out.append((plan, area))
    return out


# ----- Main pipeline ----------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)

    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".json"))
    if not files:
        raise SystemExit(f"No JSON files found in {INPUT_DIR}/ — did you run the data download?")

    # Stable shuffle so train/valid split is reproducible
    rng.shuffle(files)
    valid_files = files[:VALIDATION_HOUSES]
    train_files = files[VALIDATION_HOUSES:]

    print(f"Found {len(files)} houses → {len(train_files)} train / {len(valid_files)} valid")

    train_conversations, valid_conversations = [], []
    area_distribution = []

    for is_train, group in [(True, train_files), (False, valid_files)]:
        for filename in group:
            with open(os.path.join(INPUT_DIR, filename), encoding="utf-8") as f:
                base = json.load(f)
            base_area = sum(p.get("plocha_m2", 0) for p in base.get("prostory", []))
            area_distribution.append(base_area)
            weight = area_weight(base_area)

            for plan, area in all_variants(base):
                conv = format_conversation(plan, area, rng)
                target = train_conversations if is_train else valid_conversations
                # Area-based oversampling only on training set
                if is_train:
                    for _ in range(weight):
                        target.append(conv)
                else:
                    target.append(conv)

    rng.shuffle(train_conversations)
    rng.shuffle(valid_conversations)

    with open(os.path.join(OUTPUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_conversations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(os.path.join(OUTPUT_DIR, "valid.jsonl"), "w", encoding="utf-8") as f:
        for item in valid_conversations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Summary
    print()
    print(f"Variants per house  : {len(ROTATIONS)} rot × {len(MIRRORS)} mirror × {len(SCALES)} scale = {len(ROTATIONS)*len(MIRRORS)*len(SCALES)}")
    print(f"Area distribution   : min={min(area_distribution):.0f}  max={max(area_distribution):.0f}  "
          f"mean={sum(area_distribution)/len(area_distribution):.0f}  median={sorted(area_distribution)[len(area_distribution)//2]:.0f}")
    print(f"Train examples      : {len(train_conversations):,}  (with area-weighted oversampling)")
    print(f"Valid examples      : {len(valid_conversations):,}  (no oversampling)")
    print(f"Prompt variants     : {len(SYSTEM_PROMPTS)} system × {len(USER_PROMPTS)} user")
    print()
    print("Output: data/train.jsonl, data/valid.jsonl")


if __name__ == "__main__":
    main()
