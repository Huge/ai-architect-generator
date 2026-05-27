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


def _point_on_segment(point: list[float], seg_start: list[float], seg_end: list[float],
                      eps: float = 0.15) -> bool:
    """True if `point` lies within `eps` meters of the segment (not just its endpoints)."""
    px, py = point[0], point[1]
    sx, sy = seg_start[0], seg_start[1]
    ex, ey = seg_end[0], seg_end[1]
    dx, dy = ex - sx, ey - sy
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return abs(px - sx) < eps and abs(py - sy) < eps
    t = ((px - sx) * dx + (py - sy) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = sx + t * dx
    cy = sy + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 < eps


# ----- Cleanup functions -------------------------------------------------------

def drop_degenerate_walls(plan: dict, min_len: float = 0.05) -> dict:
    """Remove walls shorter than `min_len` meters (zero-length / collapsed walls)."""
    plan["steny"] = [w for w in plan.get("steny", []) if _wall_length(w) >= min_len]
    return plan


# Valid room types for a single-family house. Anything else is model hallucination
# (Elevator, Stairs, Apartment, etc.) and gets dropped.
VALID_ROOM_TYPES = {
    "LivingRoom", "Bedroom", "Kitchen", "Bath", "Dining", "DiningRoom",
    "Entry", "Closet", "Hall", "Hallway", "Vestibule", "Storage",
    "Garage", "Office", "Sauna", "Laundry", "Pantry", "Utility",
    "Room", "Pokoj", "Undefined",
}


def drop_outdoor_rooms(plan: dict) -> dict:
    """Remove rooms explicitly tagged as outdoor (venkovni=true)."""
    rooms = plan.get("prostory", [])
    plan["prostory"] = [
        r for r in rooms
        if not r.get("venkovni", False) and r.get("typ") != "Outdoor"
    ]
    return plan


def drop_invalid_room_types(plan: dict) -> dict:
    """Remove rooms whose `typ` isn't appropriate for a single-family house.

    Catches hallucinations like "Elevator", "Stairs", "Apartment", etc.
    """
    plan["prostory"] = [
        r for r in plan.get("prostory", [])
        if r.get("typ", "Undefined") in VALID_ROOM_TYPES
    ]
    return plan


# Czech labels for relabeled rooms.
_CZ_LABELS = {
    "LivingRoom": "Obývák",
    "Bedroom": "Ložnice",
    "Kitchen": "Kuchyně",
    "Bath": "Koupelna",
    "Entry": "Vstup",
    "Dining": "Jídelna",
    "Hall": "Hala",
    "Closet": "Šatna",
    "Storage": "Sklad",
    "Office": "Pracovna",
}


def fix_room_labels(plan: dict) -> dict:
    """Correct obviously-wrong room labels — but never create duplicate LivingRooms.

    A real house has ONE LivingRoom. If we'd be creating a second one, fall back
    to Bedroom or Dining depending on size.
    """
    rooms = plan.get("prostory", [])
    existing_types = {r.get("typ") for r in rooms}

    def pick(new_typ: str, fallback: str) -> str:
        """If new_typ is LivingRoom and we already have one, use fallback."""
        if new_typ == "LivingRoom" and "LivingRoom" in existing_types:
            return fallback
        existing_types.add(new_typ)
        return new_typ

    for r in rooms:
        area = r.get("plocha_m2", 0)
        typ = r.get("typ", "Undefined")
        new_typ = None

        if typ in ("Entry", "Vestibule", "Hall", "Hallway") and area > 15:
            new_typ = pick("LivingRoom" if area > 25 else "Dining", "Bedroom")
        elif typ == "Bath" and area > 15:
            new_typ = pick("Bedroom", "Bedroom")
        elif typ == "Closet" and area > 10:
            new_typ = pick("Bedroom" if area > 12 else "Storage", "Storage")
        elif typ == "Kitchen" and area > 40:
            new_typ = pick("LivingRoom", "Bedroom")
        elif typ == "Bedroom" and area > 45:
            new_typ = pick("LivingRoom", "Bedroom")
        elif typ in ("Room", "Pokoj", "Undefined"):
            if area >= 25:
                new_typ = pick("LivingRoom", "Bedroom")
            elif area >= 10:
                new_typ = "Bedroom"
            elif area >= 5:
                new_typ = "Dining"
            else:
                new_typ = "Bath"

        if new_typ:
            r["typ"] = new_typ
            r["nazev"] = _CZ_LABELS.get(new_typ, new_typ)

    return plan


def drop_disconnected_room_islands(plan: dict, share_threshold: float = 0.5) -> dict:
    """Keep only rooms that share at least one wall segment with another room.

    Two rooms are 'connected' if their polygon edges overlap (share a wall) for
    at least `share_threshold` meters. This catches the case where a room is
    bbox-adjacent but isn't actually sharing a wall with anyone — i.e. it's
    floating in its own little island.
    """
    rooms = [r for r in plan.get("prostory", []) if r.get("polygon")]
    if len(rooms) <= 1:
        return plan

    def edges(poly):
        return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]

    def edge_overlap_length(e1, e2, eps: float = 0.15) -> float:
        """Length of the overlap between two collinear-ish edges (0 if not collinear)."""
        (a1, a2), (b1, b2) = e1, e2
        # Both horizontal at same y
        if abs(a1[1] - a2[1]) < eps and abs(b1[1] - b2[1]) < eps and abs(a1[1] - b1[1]) < eps:
            ax1, ax2 = sorted([a1[0], a2[0]])
            bx1, bx2 = sorted([b1[0], b2[0]])
            return max(0.0, min(ax2, bx2) - max(ax1, bx1))
        # Both vertical at same x
        if abs(a1[0] - a2[0]) < eps and abs(b1[0] - b2[0]) < eps and abs(a1[0] - b1[0]) < eps:
            ay1, ay2 = sorted([a1[1], a2[1]])
            by1, by2 = sorted([b1[1], b2[1]])
            return max(0.0, min(ay2, by2) - max(ay1, by1))
        return 0.0

    room_edges = [edges(r["polygon"]) for r in rooms]
    n = len(rooms)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            for ei in room_edges[i]:
                for ej in room_edges[j]:
                    if edge_overlap_length(ei, ej) >= share_threshold:
                        union(i, j)
                        break
                else:
                    continue
                break

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    def cluster_area(indices):
        return sum(rooms[i].get("plocha_m2", 0) for i in indices)

    largest = max(clusters.values(), key=cluster_area)
    keep_ids = {rooms[i].get("id") for i in largest}

    plan["prostory"] = [r for r in plan.get("prostory", []) if r.get("id") in keep_ids]
    return plan


def drop_tiny_rooms(plan: dict, min_area: float = 1.0) -> dict:
    """Remove rooms smaller than `min_area` m² (zero / collapsed / garbage rooms).

    1.0 m² is a generous floor — typical toilets are 1.2-1.5 m². Anything smaller
    is almost certainly model noise.
    """
    plan["prostory"] = [
        r for r in plan.get("prostory", [])
        if r.get("plocha_m2", 0) >= min_area
    ]
    return plan


def _point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test (handles concave polygons)."""
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def drop_overlapping_rooms(plan: dict, size_ratio: float = 0.5) -> dict:
    """Drop a smaller room only if its centroid is inside a SIGNIFICANTLY larger room.

    Catches "bathroom polygon placed inside living-room polygon" without false-
    positives on adjacent rooms of similar size.

    A room is dropped only if BOTH:
      - its centroid lies inside another room's polygon, AND
      - that other room is at least `1/size_ratio` × bigger (default: 2× bigger)
    """
    rooms = [r for r in plan.get("prostory", []) if r.get("polygon")]
    rooms.sort(key=lambda r: r.get("plocha_m2", 0), reverse=True)
    kept = []
    for r in rooms:
        cx, cy = _polygon_centroid(r["polygon"])
        my_area = r.get("plocha_m2", 0)
        is_engulfed = False
        for k in kept:
            k_area = k.get("plocha_m2", 0)
            if k_area > 0 and (my_area / k_area) < size_ratio:
                if _point_in_polygon([cx, cy], k["polygon"]):
                    is_engulfed = True
                    break
        if not is_engulfed:
            kept.append(r)
    plan["prostory"] = kept
    return plan


def drop_orphan_rooms(
    plan: dict,
    margin: float = 0.1,
    max_outside_ratio: float = 0.20,
    hard_outside_distance: float = 1.0,
) -> dict:
    """Remove rooms whose polygon lies (mostly or significantly) outside the wall envelope.

    A room is dropped if EITHER:
      - more than `max_outside_ratio` of its vertices fall outside the envelope, OR
      - any single vertex is more than `hard_outside_distance` meters outside.

    The hard check catches rooms that "stick out" with only one vertex
    (e.g. a closet bleeding 2.5 m past the right wall).
    """
    walls = plan.get("steny", [])
    if not walls:
        return plan
    min_x, min_y, max_x, max_y = _wall_envelope(walls)

    def outside_distance(p: list[float]) -> float:
        """Distance the point lies outside the envelope (0 if inside)."""
        x, y = p[0], p[1]
        dx = max(min_x - x, x - max_x, 0.0)
        dy = max(min_y - y, y - max_y, 0.0)
        return (dx * dx + dy * dy) ** 0.5

    kept = []
    for r in plan.get("prostory", []):
        poly = r.get("polygon") or []
        if not poly:
            continue
        distances = [outside_distance(p) for p in poly]
        outside_count = sum(1 for d in distances if d > margin)
        far_out = max(distances)
        if outside_count / len(poly) <= max_outside_ratio and far_out <= hard_outside_distance:
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


def drop_floating_walls(plan: dict, eps: float = 0.15, max_passes: int = 3) -> dict:
    """Remove walls whose endpoints aren't anchored to another wall.

    An endpoint is "anchored" if:
      - it has an explicit od_constraint/do_constraint pointing at a valid wall, OR
      - it lies within `eps` meters of another wall's segment

    Runs iteratively up to `max_passes` times because dropping a wall can make
    other walls become floating in turn.
    """
    walls = plan.get("steny", [])
    if not walls:
        return plan

    for _ in range(max_passes):
        walls_by_id = {w["id"]: w for w in walls if "id" in w}

        def is_anchored(point: list[float], my_idx: int, constraint) -> bool:
            if constraint and isinstance(constraint, dict):
                host_id = constraint.get("host")
                host = walls_by_id.get(host_id)
                # Constraint host must exist AND the endpoint must actually lie
                # on/near that host wall. Otherwise we treat it as unanchored noise.
                if host and _point_on_segment(point, host["od"], host["do"], eps * 1.5):
                    return True
            for i, other in enumerate(walls):
                if i == my_idx:
                    continue
                if _point_on_segment(point, other["od"], other["do"], eps):
                    return True
            return False

        kept = []
        for i, w in enumerate(walls):
            od_anchored = is_anchored(w["od"], i, w.get("od_constraint"))
            do_anchored = is_anchored(w["do"], i, w.get("do_constraint"))
            if od_anchored and do_anchored:
                kept.append(w)

        if len(kept) == len(walls):
            break  # nothing to drop this pass — converged
        walls = kept

    plan["steny"] = walls
    return plan


def drop_extreme_length_walls(plan: dict, max_wall_length: float = 35.0, factor: float = 6.0) -> dict:
    """Remove absurdly long wall segments (spikes/stray rays).

    Keeps walls that are plausible for house-scale geometry while dropping
    pathological outliers like 80m+ rays from a 90m² floor plan.
    """
    walls = plan.get("steny", [])
    if not walls:
        return plan

    lengths = [_wall_length(w) for w in walls]
    positive = sorted(l for l in lengths if l > 0)
    if not positive:
        return plan
    median_len = positive[len(positive) // 2]
    dynamic_cap = max(max_wall_length, median_len * factor)

    plan["steny"] = [w for w in walls if _wall_length(w) <= dynamic_cap]
    return plan


def drop_walls_outside_room_bbox(plan: dict, margin: float = 0.2, outside_ratio: float = 0.45) -> dict:
    """Drop wall segments that live mostly outside the room bounding box.

    This catches long horizontal/vertical rays that shoot out from the floor plan
    while preserving walls that are near the actual room envelope.
    """
    rooms = [r for r in plan.get("prostory", []) if r.get("polygon")]
    walls = plan.get("steny", [])
    if not rooms or not walls:
        return plan

    xs, ys = [], []
    for r in rooms:
        for p in r["polygon"]:
            xs.append(p[0])
            ys.append(p[1])
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    def outside(x: float, y: float) -> bool:
        return x < min_x or x > max_x or y < min_y or y > max_y

    kept = []
    for w in walls:
        x1, y1 = w["od"]
        x2, y2 = w["do"]
        outside_count = 0
        # Uniform sampling along segment to estimate outside proportion.
        for i in range(11):
            t = i / 10.0
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            if outside(x, y):
                outside_count += 1
        frac = outside_count / 11.0
        if frac <= outside_ratio:
            kept.append(w)
    plan["steny"] = kept
    return plan


def trim_wall_overhangs(plan: dict, margin: float = 0.3) -> dict:
    """Trim wall endpoints that extend past the room bounding box.

    Instead of dropping the whole wall, this just truncates the 'od' or 'do'
    coordinates to align with the outermost room boundary. This fixes the
    long horizontal/vertical spikes without deleting valid interior walls.
    """
    rooms = [r for r in plan.get("prostory", []) if r.get("polygon")]
    walls = plan.get("steny", [])
    if not rooms or not walls:
        return plan

    xs, ys = [], []
    for r in rooms:
        for p in r["polygon"]:
            xs.append(p[0])
            ys.append(p[1])
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    kept = []
    for w in walls:
        x1, y1 = w["od"]
        x2, y2 = w["do"]
        
        # Clamp coordinates to the bounding box
        x1 = max(min_x, min(max_x, x1))
        y1 = max(min_y, min(max_y, y1))
        x2 = max(min_x, min(max_x, x2))
        y2 = max(min_y, min(max_y, y2))
        
        # If clamping collapsed the wall entirely, drop it
        if abs(x2 - x1) < 0.05 and abs(y2 - y1) < 0.05:
            continue
            
        w["od"] = [round(x1, 3), round(y1, 3)]
        w["do"] = [round(x2, 3), round(y2, 3)]
        kept.append(w)

    plan["steny"] = kept
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


def orthogonalize_walls(plan: dict, tol: float = 0.2) -> dict:
    """Force near-axis-aligned walls to be exactly horizontal/vertical.

    Many generated walls are almost axis-aligned but off by a few centimeters
    (e.g. y=1.74 vs y=1.76). Some viewers interpret this as non-intersecting
    geometry and draw long extension rays. This normalizes those walls.
    """
    for w in plan.get("steny", []):
        x1, y1 = w["od"]
        x2, y2 = w["do"]
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) <= tol and abs(dx) > abs(dy):
            y2 = y1
        elif abs(dx) <= tol and abs(dy) > abs(dx):
            x2 = x1
        w["od"] = [round(x1, 2), round(y1, 2)]
        w["do"] = [round(x2, 2), round(y2, 2)]
    return plan


def strip_wall_constraints(plan: dict) -> dict:
    """Remove wall constraint metadata from final output.

    Coordinates are already explicit after cleanup/snap/scale. Keeping stale
    od_constraint/do_constraint metadata can confuse downstream viewers that
    try to re-solve constraints and draw long construction rays.
    """
    for w in plan.get("steny", []):
        w["od_constraint"] = None
        w["do_constraint"] = None
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

def rebuild_walls_from_rooms(plan: dict, snap_eps: float = 0.1) -> dict:
    """Replace the model's walls with walls derived from room polygon edges.

    Algorithm:
      1. For each room edge, split it at any other room's vertex that lies on it
         (this handles T-junctions where two rooms meet on the same line).
      2. Snap all endpoints to a grid.
      3. Count how often each canonical edge appears across all rooms.
      4. Edges appearing once → exterior wall (obvodova).
         Edges appearing twice → interior partition (pricka).
         Edges appearing more (rare) → still interior.

    Guarantees a watertight floor plan when room polygons are valid.
    """
    rooms = [r for r in plan.get("prostory", []) if r.get("polygon")]
    if not rooms:
        return plan

    def snap_pt(p):
        return (round(p[0] / snap_eps) * snap_eps, round(p[1] / snap_eps) * snap_eps)

    # Collect every vertex from every polygon (we'll use these to split edges at T-junctions)
    all_vertices: set[tuple] = set()
    for r in rooms:
        for p in r["polygon"]:
            all_vertices.add(snap_pt(p))

    def split_edge_at_collinear_vertices(a, b):
        """Return list of sub-edges if any vertices lie strictly between a and b."""
        sa, sb = snap_pt(a), snap_pt(b)
        if sa == sb:
            return []
        ax, ay = sa
        bx, by = sb
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-6:
            return []
        # Find vertices strictly between a and b on this line
        intermediate = []
        for v in all_vertices:
            if v == sa or v == sb:
                continue
            vx, vy = v
            # Cross product near zero = collinear
            cross = (vx - ax) * dy - (vy - ay) * dx
            if abs(cross) > snap_eps * 0.5:
                continue
            # Parameter t along the line; must be in (0, 1) for the point to be strictly between
            t = ((vx - ax) * dx + (vy - ay) * dy) / length_sq
            if 0.0 + 1e-6 < t < 1.0 - 1e-6:
                intermediate.append((t, v))
        # Build the sub-edges
        intermediate.sort()
        points = [sa] + [v for _, v in intermediate] + [sb]
        return [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    def canon_edge(a, b):
        if a == b:
            return None
        return (a, b) if a < b else (b, a)

    # Count canonical edges with splitting
    edge_count: dict[tuple, int] = {}
    for r in rooms:
        poly = r["polygon"]
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            for sub_a, sub_b in split_edge_at_collinear_vertices(a, b):
                key = canon_edge(sub_a, sub_b)
                if key is None:
                    continue
                edge_count[key] = edge_count.get(key, 0) + 1

    new_walls = []
    wall_idx = 1
    for (a, b), count in edge_count.items():
        if abs(a[0] - b[0]) < 1e-3 and abs(a[1] - b[1]) < 1e-3:
            continue
        is_exterior = count == 1
        wall = {
            "id": f"W{wall_idx}",
            "od": [round(a[0], 2), round(a[1], 2)],
            "do": [round(b[0], 2), round(b[1], 2)],
            "tloustka": 0.3 if is_exterior else 0.15,
            "typ": "obvodova" if is_exterior else "pricka",
            "od_constraint": None,
            "do_constraint": None,
        }
        new_walls.append(wall)
        wall_idx += 1

    plan["steny"] = new_walls
    plan["otvory"] = []  # openings referenced old walls; drop them
    return plan


def has_closed_exterior_loop(plan: dict, eps: float = 0.15) -> bool:
    """
    Validates if the walls in the floor plan form at least one fully closed loop.
    
    Converts wall segments into a graph and checks the largest connected 
    component. If the outermost structure has dead ends (nodes with degree < 2), 
    or doesn't contain a cycle, the house is not "watertight".
    """
    walls = plan.get("steny", [])
    if not walls:
        return False

    adj: dict[tuple[float, float], set[tuple[float, float]]] = {}
    
    def _get_key(pt: list[float]) -> tuple[float, float]:
        return (round(pt[0], 2), round(pt[1], 2))

    for w in walls:
        p1 = _get_key(w["od"])
        p2 = _get_key(w["do"])
        
        if p1 not in adj: adj[p1] = set()
        if p2 not in adj: adj[p2] = set()
        
        if p1 != p2:
            adj[p1].add(p2)
            adj[p2].add(p1)

    if not adj:
        return False

    visited = set()
    components = []
    
    for node in adj:
        if node not in visited:
            comp = set()
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    comp.add(curr)
                    stack.extend(adj[curr] - visited)
            components.append(comp)
            
    largest_comp = max(components, key=len)
    
    xs = [pt[0] for pt in largest_comp]
    ys = [pt[1] for pt in largest_comp]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    margin = eps * 2
    
    for node in largest_comp:
        x, y = node
        is_on_edge = (
            abs(x - min_x) < margin or 
            abs(x - max_x) < margin or 
            abs(y - min_y) < margin or 
            abs(y - max_y) < margin
        )
        
        if is_on_edge and len(adj[node]) < 2:
            return False

    def contains_cycle(start_node: tuple[float, float]) -> bool:
        stack = [(start_node, None)]
        local_visited = set()
        
        while stack:
            curr, parent = stack.pop()
            if curr in local_visited:
                return True
            
            local_visited.add(curr)
            
            for neighbor in adj[curr]:
                if neighbor != parent:
                    if neighbor in local_visited:
                        return True
                    stack.append((neighbor, curr))
        return False

    return contains_cycle(next(iter(largest_comp)))


def post_process(plan: dict, target_area: Optional[float] = None) -> dict:
    """Run the full cleanup pipeline on a plan.

    Args:
        plan: parsed JSON dict with keys 'steny', 'otvory', 'prostory'.
        target_area: if provided, rescale so total interior area matches this value (m²).

    Returns:
        The cleaned plan (mutated in place AND returned for convenience).
    """
    # === Phase 1: clean up the ROOMS (we'll rebuild walls from them) ===
    plan = drop_outdoor_rooms(plan)
    plan = drop_invalid_room_types(plan)
    plan = recompute_areas(plan)
    plan = drop_tiny_rooms(plan)
    plan = drop_overlapping_rooms(plan)
    plan = drop_disconnected_room_islands(plan)
    plan = close_polygons(plan)
    plan = recompute_areas(plan)

    # === Phase 2: rescale to target area BEFORE rebuilding walls ===
    if target_area is not None and target_area > 0:
        plan = rescale_to_target_area(plan, target_area)
        plan = recompute_areas(plan)

    # === Phase 3: rebuild walls deterministically from room polygons ===
    # This guarantees watertight geometry and drops the model's messy walls entirely.
    plan = rebuild_walls_from_rooms(plan)

    # === Phase 4: final label cleanup ===
    plan = fix_room_labels(plan)
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
