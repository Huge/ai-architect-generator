"""Geometric post-processor for Kalkulio floor plans.

Cleans up sloppy LLM output into a competition-ready floor plan:
    1. Drops outdoor / "venkovni" rooms
    2. Drops rooms whose centroid lies outside the wall envelope
    3. Snaps wall endpoints that are within ~10 cm of each other
    4. Drops openings whose host wall is missing or whose position is invalid
    5. Recomputes plocha_m2 from polygon vertices (shoelace)
    6. Rescales the whole plan so total interior area matches the user's request

Pure Python (no numpy/shapely required).

CLI:
    python post_process.py INPUT.json [--target-area 120] [-o OUTPUT.json]
    python post_process.py outputs/sample_180m2.json --target-area 180

Library:
    from post_process import post_process
    cleaned = post_process(plan, target_area=180.0)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


# ----- Helpers -----------------------------------------------------------------

def _polygon_area(polygon: list[list[float]]) -> float:
    """Shoelace formula. Returns positive area in m²."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _polygon_centroid(polygon: list[list[float]]) -> tuple[float, float]:
    n = len(polygon)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return cx, cy


def _wall_envelope(walls: list[dict]) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of all wall endpoints. (min_x, min_y, max_x, max_y)."""
    xs, ys = [], []
    for w in walls:
        xs.extend([w["od"][0], w["do"][0]])
        ys.extend([w["od"][1], w["do"][1]])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _wall_length(w: dict) -> float:
    dx = w["do"][0] - w["od"][0]
    dy = w["do"][1] - w["od"][1]
    return (dx * dx + dy * dy) ** 0.5


# ----- Cleanup functions -------------------------------------------------------

def drop_outdoor_rooms(plan: dict) -> dict:
    """Remove rooms explicitly tagged as outdoor (venkovni=true)."""
    rooms = plan.get("prostory", [])
    plan["prostory"] = [
        r for r in rooms
        if not r.get("venkovni", False) and r.get("typ") != "Outdoor"
    ]
    return plan


def drop_orphan_rooms(plan: dict, margin: float = 0.1, max_outside_ratio: float = 0.25) -> dict:
    """Remove rooms whose polygon lies (mostly) outside the wall envelope.

    A room is dropped if more than `max_outside_ratio` of its vertices fall outside
    the wall bounding box (extended by `margin` to allow for small rounding noise).
    This catches both fully-orphan rooms and ones that bleed across the outer wall.
    """
    walls = plan.get("steny", [])
    if not walls:
        return plan
    min_x, min_y, max_x, max_y = _wall_envelope(walls)

    def is_outside(p: list[float]) -> bool:
        x, y = p[0], p[1]
        return (
            x < (min_x - margin)
            or x > (max_x + margin)
            or y < (min_y - margin)
            or y > (max_y + margin)
        )

    kept = []
    for r in plan.get("prostory", []):
        poly = r.get("polygon") or []
        if not poly:
            continue
        outside_count = sum(1 for p in poly if is_outside(p))
        if outside_count / len(poly) <= max_outside_ratio:
            kept.append(r)
    plan["prostory"] = kept
    return plan


def snap_wall_endpoints(plan: dict, eps: float = 0.15) -> dict:
    """Snap wall endpoints that are within `eps` meters of each other to a shared point.

    Closes small gaps where the model generated 'almost-touching' walls.
    """
    walls = plan.get("steny", [])
    if not walls:
        return plan

    # Collect every endpoint as (wall_idx, "od"|"do")
    endpoints: list[tuple[int, str, float, float]] = []
    for i, w in enumerate(walls):
        endpoints.append((i, "od", w["od"][0], w["od"][1]))
        endpoints.append((i, "do", w["do"][0], w["do"][1]))

    # Union-Find clustering of nearby endpoints
    n = len(endpoints)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        _, _, xi, yi = endpoints[i]
        for j in range(i + 1, n):
            _, _, xj, yj = endpoints[j]
            if abs(xi - xj) <= eps and abs(yi - yj) <= eps:
                union(i, j)

    # For each cluster, compute the average (snapped) point
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    snapped: dict[int, tuple[float, float]] = {}
    for root, members in clusters.items():
        cx = round(sum(endpoints[m][2] for m in members) / len(members), 2)
        cy = round(sum(endpoints[m][3] for m in members) / len(members), 2)
        for m in members:
            snapped[m] = (cx, cy)

    # Apply snapped coordinates back onto the walls
    for i, w in enumerate(walls):
        ox, oy = snapped[2 * i]
        dx, dy = snapped[2 * i + 1]
        w["od"] = [ox, oy]
        w["do"] = [dx, dy]

    return plan


def drop_disconnected_openings(plan: dict, tol: float = 0.1) -> dict:
    """Remove windows/doors whose host wall is missing or whose position overflows the wall."""
    walls_by_id = {w["id"]: w for w in plan.get("steny", []) if "id" in w}
    kept = []
    for o in plan.get("otvory", []):
        wall_id = o.get("stena")
        if wall_id not in walls_by_id:
            continue
        wlen = _wall_length(walls_by_id[wall_id])
        pos = o.get("pozice", 0) or 0
        width = o.get("sirka", 0) or 0
        if pos < -tol or (pos + width) > (wlen + tol):
            continue
        kept.append(o)
    plan["otvory"] = kept
    return plan


def close_polygons(plan: dict) -> dict:
    """Ensure each room polygon starts and ends at the same point."""
    for r in plan.get("prostory", []):
        poly = r.get("polygon") or []
        if len(poly) >= 3 and poly[0] != poly[-1]:
            poly.append([poly[0][0], poly[0][1]])
            r["polygon"] = poly
    return plan


def recompute_areas(plan: dict) -> dict:
    """Recompute plocha_m2 from polygon vertices using the shoelace formula."""
    for r in plan.get("prostory", []):
        if r.get("polygon"):
            r["plocha_m2"] = round(_polygon_area(r["polygon"]), 2)
    return plan


def rescale_to_target_area(plan: dict, target_area: float) -> dict:
    """Uniformly scale all coordinates so total interior area equals `target_area`.

    Only interior (non-venkovni) rooms count toward the total.
    """
    interior = [r for r in plan.get("prostory", []) if not r.get("venkovni", False)]
    actual = sum(r.get("plocha_m2", 0) for r in interior)
    if actual <= 0 or target_area <= 0:
        return plan

    scale = (target_area / actual) ** 0.5

    for w in plan.get("steny", []):
        w["od"] = [round(c * scale, 2) for c in w["od"]]
        w["do"] = [round(c * scale, 2) for c in w["do"]]
        if "tloustka" in w and w["tloustka"] is not None:
            w["tloustka"] = round(w["tloustka"] * scale, 3)

    for o in plan.get("otvory", []):
        if "pozice" in o and o["pozice"] is not None:
            o["pozice"] = round(o["pozice"] * scale, 2)
        if "sirka" in o and o["sirka"] is not None:
            o["sirka"] = round(o["sirka"] * scale, 2)

    for r in plan.get("prostory", []):
        if r.get("polygon"):
            r["polygon"] = [[round(c * scale, 2) for c in p] for p in r["polygon"]]
        if "plocha_m2" in r:
            r["plocha_m2"] = round(r["plocha_m2"] * scale * scale, 2)

    return plan


# ----- Main orchestrator -------------------------------------------------------

def post_process(plan: dict, target_area: Optional[float] = None) -> dict:
    """Run the full cleanup pipeline on a plan.

    Args:
        plan: parsed JSON dict with keys 'steny', 'otvory', 'prostory'.
        target_area: if provided, rescale so total interior area matches this value (m²).

    Returns:
        The cleaned plan (mutated in place AND returned for convenience).
    """
    plan = drop_outdoor_rooms(plan)
    plan = snap_wall_endpoints(plan)
    plan = drop_orphan_rooms(plan)
    plan = drop_disconnected_openings(plan)
    plan = close_polygons(plan)
    plan = recompute_areas(plan)
    if target_area is not None and target_area > 0:
        plan = rescale_to_target_area(plan, target_area)
        plan = recompute_areas(plan)
    return plan


# ----- CLI ---------------------------------------------------------------------

def _summarize(plan: dict, label: str) -> None:
    n_walls = len(plan.get("steny", []))
    n_openings = len(plan.get("otvory", []))
    n_rooms = len(plan.get("prostory", []))
    total_area = round(
        sum(r.get("plocha_m2", 0) for r in plan.get("prostory", []) if not r.get("venkovni", False)),
        2,
    )
    print(f"  {label:8s} walls={n_walls:3d}  openings={n_openings:3d}  rooms={n_rooms:3d}  total_area={total_area} m²")


def _main() -> int:
    p = argparse.ArgumentParser(description="Clean up a Kalkulio floor plan JSON.")
    p.add_argument("input", help="Path to raw JSON file (model output)")
    p.add_argument("-o", "--output", help="Path to write cleaned JSON (default: alongside input, with _clean suffix)")
    p.add_argument("--target-area", type=float, default=None, help="Rescale so total interior area = this (m²)")
    p.add_argument("--pretty", action="store_true", help="Write indented JSON instead of compact")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    with open(args.input, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"ERROR: input is not valid JSON ({e})", file=sys.stderr)
        return 3

    print(f"Processing {args.input}")
    _summarize(plan, "before")

    cleaned = post_process(plan, target_area=args.target_area)
    _summarize(cleaned, "after")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_clean{ext}"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        else:
            json.dump(cleaned, f, separators=(",", ":"), ensure_ascii=False)
    print(f"  Wrote {args.output} ({os.path.getsize(args.output)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
