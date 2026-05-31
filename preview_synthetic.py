"""Quick visual preview of a synthetic plan to verify quality.

Usage:
    python preview_synthetic.py raw_data/synthetic_001.json
    python preview_synthetic.py raw_data/synthetic_001.json --save preview.png
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOM_COLORS = {
    "LivingRoom": "#ffe4b5", "Bedroom": "#e6e6fa", "Kitchen": "#ffdab9",
    "Bath": "#e0ffff", "Entry": "#f5f5dc", "Dining": "#fffacd",
    "Hall": "#f0f8ff", "Hallway": "#f0f8ff", "Vestibule": "#f5f5dc",
    "Closet": "#f5fffa", "Storage": "#f5fffa", "Garage": "#dcdcdc",
    "Office": "#fff0f5", "Sauna": "#ffe4e1", "Laundry": "#e0f7fa",
    "Pantry": "#fff8dc", "Utility": "#f5fffa",
}


def preview(path: str, save: str = None) -> None:
    plan = json.loads(open(path).read())
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("white")

    rooms = plan.get("prostory", [])
    walls = plan.get("steny", [])

    for r in rooms:
        poly = r.get("polygon")
        if not poly:
            continue
        color = ROOM_COLORS.get(r.get("typ", ""), "#f0f0f0")
        ax.add_patch(Polygon(poly, closed=True, facecolor=color,
                             edgecolor="#cccccc", linewidth=0.5, alpha=0.85, zorder=1))

    for w in walls:
        x1, y1 = w["od"]
        x2, y2 = w["do"]
        is_ext = w.get("typ") in ("obvodova", "nosna")
        lw = 3.5 if is_ext else 2.0
        ax.plot([x1, x2], [y1, y2], color="#2c3e50", linewidth=lw,
                solid_capstyle="round", zorder=3)

    for r in rooms:
        poly = r.get("polygon")
        if not poly:
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        label = r.get("nazev") or r.get("typ", "?")
        area = r.get("plocha_m2", 0)
        ax.text(cx, cy, f"{label}\n{area:.1f} m²", ha="center", va="center",
                fontsize=10, color="#2c3e50",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc",
                          boxstyle="round,pad=0.3"), zorder=10)

    total = sum(r.get("plocha_m2", 0) for r in rooms)
    n_rooms = len([r for r in rooms if r.get("polygon")])
    ax.set_title(f"{total:.0f} m²  •  {n_rooms} rooms",
                 fontsize=13, color="#2c3e50")
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.05)
    ax.axis("off")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=120, bbox_inches="tight")
        print(f"Saved to {save}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to synthetic_NNN.json")
    parser.add_argument("--save", help="Save to PNG instead of showing")
    args = parser.parse_args()
    preview(args.path, args.save)
