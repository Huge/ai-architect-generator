"""Generate synthetic Kalkulio-format floor plans using Gemini 2.5 Pro.

Approach:
  1. Ask Gemini to design just ROOMS (polygons + type + area + Czech name)
  2. Let our existing post_process.py rebuild walls deterministically
  3. Validate watertightness; keep only the good ones
  4. Save to raw_data/synthetic_*.json so prepare_data.py picks them up

Usage:
    export GEMINI_API_KEY="..."
    python generate_synthetic_data.py --target 100 --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

from post_process import has_closed_exterior_loop, post_process

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ google-genai not installed. Run: pip install google-genai")
    sys.exit(1)


# ----- Configuration ---------------------------------------------------------

# Model fallback chain: try the first; if its daily quota is exhausted,
# automatically move to the next. Each Gemini model has its own quota pool
# so we can get ~5x the effective free-tier capacity this way.
DEFAULT_MODELS = [
    "gemini-3.5-flash",          # Best quality, lowest quota
    "gemini-3.1-pro-preview",    # Strong Pro tier, separate quota
    "gemini-3-flash-preview",    # Older Flash preview, separate quota
    "gemini-2.5-flash",          # Legacy Flash, generous quota
    "gemini-2.5-flash-lite",     # Fastest fallback, highest quota
]
OUTPUT_DIR = Path("raw_data")
REFERENCE_FILE = Path("kalkulio_all.json")
NUM_FEW_SHOT = 2  # Few-shot examples per prompt

# Target distribution of plans
AREA_BUCKETS = [
    (60, 80, 25),    # small houses
    (80, 110, 35),   # medium houses
    (110, 150, 25),  # large houses
    (150, 200, 15),  # very large houses
]

STYLES = [
    "compact rectangular layout",
    "L-shaped layout with a corner",
    "open-plan kitchen and living room",
    "narrow long layout",
    "traditional layout with central hallway",
    "modern open layout",
    "U-shaped layout",
    "rectangular ranch-style layout",
]

ROOM_HINTS = {
    "small": "with 3-5 rooms: Entry, Kitchen, LivingRoom, Bath, and 1 Bedroom",
    "medium": "with 5-7 rooms: Entry, Kitchen, LivingRoom, Bath, 1-2 Bedrooms, and optionally Dining or Storage",
    "large": "with 6-9 rooms: Entry, Kitchen, LivingRoom, Bath, 2-3 Bedrooms, Dining, Closet, Office or Storage",
    "xlarge": "with 8-12 rooms: Entry, Kitchen, LivingRoom, Dining, 2 Baths, 3-4 Bedrooms, Closet, Office, Laundry, Storage",
}


# ----- Schema for Gemini structured output -----------------------------------

ROOM_SCHEMA = {
    "type": "object",
    "required": ["typ", "nazev", "polygon", "plocha_m2"],
    "properties": {
        "typ": {
            "type": "string",
            "enum": [
                "LivingRoom", "Bedroom", "Kitchen", "Bath", "Dining",
                "Entry", "Hall", "Hallway", "Closet", "Storage",
                "Office", "Laundry", "Pantry", "Vestibule",
            ],
        },
        "nazev": {
            "type": "string",
            "description": "Czech name for the room (e.g. Obývák, Ložnice, Kuchyně, Koupelna, Vstup, Jídelna, Šatna, Sklad, Pracovna)",
        },
        "polygon": {
            "type": "array",
            "description": "Closed polygon as list of [x, y] points in meters, in counter-clockwise order. Must be rectangular (4 corners) or rectilinear (all angles 90°).",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "minItems": 4,
        },
        "plocha_m2": {
            "type": "number",
            "description": "Area of this room in square meters",
        },
    },
}

PLAN_SCHEMA = {
    "type": "object",
    "required": ["prostory"],
    "properties": {
        "prostory": {
            "type": "array",
            "description": "List of rooms forming the floor plan. ALL rooms must tile together with no gaps and no overlaps, forming a single connected house outline.",
            "items": ROOM_SCHEMA,
            "minItems": 3,
        }
    },
}


# ----- Few-shot example builder ----------------------------------------------

def load_reference_plans() -> list[dict]:
    """Load reference plans from kalkulio_all.json, keep only the small/clean ones."""
    if not REFERENCE_FILE.exists():
        print(f"❌ Reference file {REFERENCE_FILE} not found")
        sys.exit(1)
    raw = json.loads(REFERENCE_FILE.read_text())
    candidates = []
    for h in raw:
        rooms = h.get("prostory", [])
        # Keep only houses with reasonable room counts and known types
        if 4 <= len(rooms) <= 12:
            candidates.append(h)
    print(f"Loaded {len(candidates)} reference houses (filtered from {len(raw)})")
    return candidates


def simplify_for_fewshot(plan: dict) -> dict:
    """Strip a Kalkulio plan to just the room info Gemini needs to see."""
    rooms = []
    for r in plan.get("prostory", []):
        typ = r.get("typ", "Undefined")
        if typ in ("Undefined", "Outdoor", "Basement"):
            continue
        rooms.append({
            "typ": typ,
            "nazev": r.get("nazev", typ),
            "polygon": [[round(p[0], 2), round(p[1], 2)] for p in r["polygon"]],
            "plocha_m2": round(r.get("plocha_m2", 0), 2),
        })
    return {"prostory": rooms}


# ----- Prompt construction ---------------------------------------------------

SYSTEM_PROMPT = """You are an expert architect specializing in single-family floor plans.

Your task is to design realistic floor plans output as JSON.

Rules:
  1. Output ONLY a JSON object with a single key "prostory" (list of rooms).
  2. Each room has: typ (English type), nazev (Czech name), polygon, plocha_m2.
  3. Polygons are rectilinear (all angles 90°), in METERS, counter-clockwise order.
  4. Rooms MUST tile together with NO gaps and NO overlaps — they form one
     connected house outline.
  5. Use realistic dimensions: bedrooms 9-18 m², living rooms 18-40 m²,
     kitchens 8-20 m², bathrooms 4-10 m², entry 4-10 m².
  6. Coordinates should be small positive numbers (e.g. start at 0,0).
  7. The plan must be WATERTIGHT — adjacent rooms must share exact edge coordinates.
"""


def build_user_prompt(area: int, style: str, room_hint: str, fewshot: list[dict]) -> str:
    fewshot_blocks = ""
    for i, ex in enumerate(fewshot, 1):
        compact = json.dumps(ex, ensure_ascii=False, separators=(",", ":"))
        fewshot_blocks += f"\n### Example {i}:\n```json\n{compact}\n```\n"

    return f"""Generate a {area} m² single-family house floor plan.

Style: {style}
Room layout: {room_hint}
Target total area: {area} m² (sum of all room areas, ±10%)

CRITICAL: Adjacent rooms MUST share exact edge coordinates so the plan is watertight.
For example, if Room A has a vertex at (5.0, 3.0), Room B sharing that wall must
also have a vertex at exactly (5.0, 3.0).
{fewshot_blocks}
Now generate ONE new floor plan as JSON. Do not copy the examples — design a fresh layout."""


# ----- Generation logic ------------------------------------------------------

# Shared model state. Models that hit DAILY quota exhaustion get burned
# permanently for this run; models with only per-minute throttling get a
# temporary cooldown.
_burned_models: set[str] = set()        # Daily quota dead — skip permanently
_model_cooldown_until: dict[str, float] = {}  # Per-minute throttle — wait, then retry


def _pick_next_model(chain: list[str]) -> Optional[str]:
    """Pick the next non-burned model, or None if all are burned."""
    for m in chain:
        if m not in _burned_models:
            return m
    return None


async def _wait_for_model(model: str) -> None:
    cooldown_until = _model_cooldown_until.get(model, 0)
    delay = cooldown_until - time.time()
    if delay > 0:
        await asyncio.sleep(delay)


async def generate_one(
    client,
    model_chain: list[str],
    area: int,
    style: str,
    room_hint: str,
    fewshot: list[dict],
    semaphore: asyncio.Semaphore,
    per_model_retries: int = 3,
) -> tuple[int, str, Optional[dict]]:
    """Generate a single plan, walking the model fallback chain on quota errors.

    Per model: retry per_model_retries times with exponential backoff.
    Across models: if a model exhausts retries, mark it burned and move on.
    Returns (area, style, plan_or_None).
    """
    async with semaphore:
        prompt = build_user_prompt(area, style, room_hint, fewshot)

        while True:
            model_name = _pick_next_model(model_chain)
            if model_name is None:
                print(f"  ✗ All models burned (area {area} m²)")
                return area, style, None

            await _wait_for_model(model_name)
            backoff = 8.0
            for attempt in range(per_model_retries):
                try:
                    response = await client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            response_schema=PLAN_SCHEMA,
                            temperature=0.9,
                            max_output_tokens=8192,
                        ),
                    )
                    plan = json.loads(response.text)
                    plan["__model__"] = model_name  # stash for metadata
                    return area, style, plan
                except Exception as e:
                    msg = str(e)
                    is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
                    if not is_rate_limit:
                        # Non-quota error — return None so caller logs it
                        print(f"  ⚠️  {model_name} error ({area} m²): {type(e).__name__}: {msg[:100]}")
                        return area, style, None
                    if attempt < per_model_retries - 1:
                        _model_cooldown_until[model_name] = time.time() + backoff
                        print(f"  ⏳ {model_name} rate-limited (attempt {attempt+1}/{per_model_retries}), waiting {backoff:.0f}s…")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 1.8, 60.0)
                        continue

            # Exhausted retries on this model → burn it and fall through to next
            _burned_models.add(model_name)
            remaining = [m for m in model_chain if m not in _burned_models]
            print(f"  🔥 {model_name} quota exhausted, switching to fallback. Remaining: {remaining}")


def validate_plan(plan: dict, target_area: int) -> tuple[bool, str, dict, str]:
    """Run through post_process and check the result is usable.

    Returns (is_valid, reason, processed_plan, model_name).
    """
    model_name = plan.pop("__model__", "unknown") if isinstance(plan, dict) else "unknown"
    if not isinstance(plan, dict) or "prostory" not in plan:
        return False, "no prostory key", {}, model_name
    if len(plan["prostory"]) < 3:
        return False, f"too few rooms ({len(plan['prostory'])})", {}, model_name

    full = {"steny": [], "otvory": [], "prostory": plan["prostory"]}

    try:
        cleaned = post_process(full, target_area=float(target_area))
    except Exception as e:
        return False, f"post_process error: {type(e).__name__}: {str(e)[:60]}", {}, model_name

    rooms = cleaned.get("prostory", [])
    walls = cleaned.get("steny", [])

    if len(rooms) < 3:
        return False, f"only {len(rooms)} rooms survived post_process", {}, model_name
    if not walls:
        return False, "no walls after rebuild", {}, model_name
    if not has_closed_exterior_loop(cleaned):
        return False, "not watertight", {}, model_name

    total_area = sum(r.get("plocha_m2", 0) for r in rooms)
    if abs(total_area - target_area) / target_area > 0.20:
        return False, f"area mismatch ({total_area:.1f} vs target {target_area})", {}, model_name

    return True, "ok", cleaned, model_name


# ----- Main loop -------------------------------------------------------------

async def main(target: int, concurrency: int, api_key: str, model_chain: list[str]):
    OUTPUT_DIR.mkdir(exist_ok=True)
    existing = sorted(OUTPUT_DIR.glob("synthetic_*.json"))
    start_idx = len(existing) + 1
    print(f"📁 Output dir: {OUTPUT_DIR}  (existing synthetic: {len(existing)})")
    print(f"🤖 Model chain (fallback order): {model_chain}")

    client = genai.Client(api_key=api_key)
    references = load_reference_plans()
    rng = random.Random(42)

    # Build the work queue
    work = []
    for lo, hi, count in AREA_BUCKETS:
        bucket_count = int(count / 100 * target)
        for _ in range(bucket_count):
            area = rng.randint(lo, hi)
            style = rng.choice(STYLES)
            if area < 80:
                hint = ROOM_HINTS["small"]
            elif area < 110:
                hint = ROOM_HINTS["medium"]
            elif area < 150:
                hint = ROOM_HINTS["large"]
            else:
                hint = ROOM_HINTS["xlarge"]
            fewshot = [simplify_for_fewshot(rng.choice(references)) for _ in range(NUM_FEW_SHOT)]
            work.append((area, style, hint, fewshot))

    rng.shuffle(work)
    print(f"🎯 Generating {len(work)} plans with concurrency={concurrency}\n")

    semaphore = asyncio.Semaphore(concurrency)
    saved = 0
    failed = 0
    failure_reasons: dict[str, int] = {}
    start = time.time()

    tasks = [
        asyncio.create_task(generate_one(client, model_chain, area, style, hint, fewshot, semaphore))
        for (area, style, hint, fewshot) in work
    ]

    model_counts: dict[str, int] = {}

    i = 0
    for fut in asyncio.as_completed(tasks):
        i += 1
        area, style, plan = await fut
        if plan is None:
            failed += 1
            failure_reasons["api_error_or_all_models_burned"] = failure_reasons.get("api_error_or_all_models_burned", 0) + 1
            continue

        ok, reason, cleaned, model_used = validate_plan(plan, area)
        if not ok:
            failed += 1
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            print(f"  [{i:3d}/{len(work)}] ✗ {area:>3d} m²  {style[:30]:30s}  [{model_used}] {reason}")
            continue

        idx = start_idx + saved
        out_path = OUTPUT_DIR / f"synthetic_{idx:03d}.json"
        cleaned["metadata"] = {
            "source": "synthetic_gemini",
            "model": model_used,
            "target_area_m2": area,
            "style": style,
        }
        out_path.write_text(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")))
        saved += 1
        model_counts[model_used] = model_counts.get(model_used, 0) + 1
        elapsed = time.time() - start
        rate = saved / elapsed if elapsed > 0 else 0
        print(f"  [{i:3d}/{len(work)}] ✓ {area:>3d} m²  {style[:30]:30s}  [{model_used}]  → {out_path.name}  ({rate:.1f}/s)")

    print()
    print("=" * 70)
    print(f"📊 Saved {saved}/{len(work)} plans ({saved/max(1,len(work))*100:.1f}% success)")
    print(f"   Time: {time.time()-start:.1f}s")
    if model_counts:
        print(f"   Plans per model:")
        for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
            print(f"     {c:>3d}  {m}")
    if _burned_models:
        print(f"   🔥 Burned models (quota exhausted): {sorted(_burned_models)}")
    if failure_reasons:
        print(f"   Failure breakdown:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"     {count:>3d}  {reason}")
    print(f"📁 Files: {OUTPUT_DIR}/synthetic_*.json")
    print()
    print("Next steps:")
    print("  1. python prepare_data.py     # rebuild train/valid jsonl with new data")
    print("  2. python train.py            # retrain (will be v4 if you bump OUTPUT_DIR)")


def cli():
    parser = argparse.ArgumentParser(description="Generate synthetic Kalkulio floor plans via Gemini")
    parser.add_argument("--target", type=int, default=100,
                        help="Total number of plans to generate (default: 100)")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Concurrent Gemini calls (default: 2; bump higher with paid tier)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model fallback chain in priority order. When the first model "
                             "exhausts its daily quota, we automatically move to the next. "
                             f"Default: {' '.join(DEFAULT_MODELS)}")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Export it first:")
        print("   export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    try:
        asyncio.run(main(args.target, args.concurrency, api_key, args.models))
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted. Any plans saved so far are in raw_data/")


if __name__ == "__main__":
    cli()
