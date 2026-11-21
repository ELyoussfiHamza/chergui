"""Single-file HTML report for a mission: the locked test, every attempt on one map, rules x attempts."""

from __future__ import annotations

import json
from pathlib import Path

from .vehicle import offset_to_latlon

TEMPLATE = Path(__file__).with_name("mission_template.html")


def build(mission_dir: Path) -> Path:
    mission_dir = Path(mission_dir)
    data = json.loads((mission_dir / "mission.json").read_text())
    place, task = data["place"], data["test"]["task"]
    ll = lambda n, e: offset_to_latlon(place["lat"], place["lon"], n, e)  # noqa: E731
    if task["type"] == "survey":
        (cn, ce), (ln, wd) = task["area_center_ne_m"], task["area_size_m"]
        data["area_latlon"] = [ll(cn + sn * ln / 2, ce + se * wd / 2) for sn, se in ((-1, -1), (-1, 1), (1, 1), (1, -1))]
    else:
        data["target_latlon"] = ll(*task["target_ne_m"])
    blob = json.dumps(data, default=float).replace("</", "<\\/")
    out = mission_dir / "mission.html"
    out.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", blob))
    return out
