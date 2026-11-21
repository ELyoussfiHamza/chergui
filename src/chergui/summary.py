"""AI incident write-up. The model explains; it never judges.

Pass/fail comes from analyze.py's fixed rules. The model only gets the measured numbers and the
event timeline, and is told to cite them, so the write-up can't invent a verdict or a figure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import config

INSTRUCTIONS = """You are a flight-test engineer writing the incident summary for a simulated drone test.
You receive JSON with: the scenario, deterministic pass/fail verdicts, metrics, fault windows and a
timestamped event timeline (t = seconds since mission start). Ground truth comes from the simulator.

Rules:
- Use ONLY numbers present in the JSON. Round sensibly. Never invent values, causes you can't support, or new verdicts.
- The verdicts are final; explain them, don't re-judge them.
- Be concrete and technical but readable by a non-specialist. No hype, no emojis.
- Write plain words, never raw field names (say "largest route deviation", not "max_deviation_m").
- Distinguish what the drone believed (estimate) from what really happened (ground truth).
- Recommendations must be physically sensible: without GNSS the drone cannot navigate to a position
  (no return-to-home, no waypoint flight); options are things like holding attitude/altitude, descending,
  alternative aiding (optical flow, vision), or changing failsafe timing and post-recovery behaviour.

Reply with a JSON object:
{
  "headline": "one sentence, max 20 words",
  "narrative": ["2-4 short paragraphs telling the flight in order, citing times (t=...) and numbers"],
  "findings": ["3-5 bullet findings, most important first"],
  "recommendations": ["2-4 concrete next steps (parameters to review, scenarios to add)"]
}"""


def _rounded(x):
    """Round floats so the model quotes '110.5 m', not '110.45684880392089 m'."""
    if isinstance(x, float):
        return round(x, 1) if abs(x) >= 1 else round(x, 2)
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    return x


def _payload(analysis: dict) -> dict:
    keep = ("scenario", "run", "passed", "verdicts", "metrics", "faults", "events")
    return _rounded({k: analysis[k] for k in keep})


def summarize(analysis: dict, run_dir: Path) -> dict | None:
    """Returns the summary dict (cached in run_dir/summary.json), or None if no API key is set."""
    cache = Path(run_dir) / "summary.json"
    if cache.exists():
        return json.loads(cache.read_text())
    if not os.getenv("OPENAI_API_KEY"):
        return None

    from openai import OpenAI

    client = OpenAI()
    resp = client.responses.create(
        model=config.OPENAI_MODEL,
        instructions=INSTRUCTIONS,
        # JSON mode requires the word "json" in the input itself, not only in the instructions.
        input="Flight test data as JSON. Reply with the JSON object described.\n"
              + json.dumps(_payload(analysis), default=float),
        text={"format": {"type": "json_object"}},
    )
    try:
        summary = json.loads(resp.output_text)
    except json.JSONDecodeError:
        summary = {"headline": "AI summary", "narrative": [resp.output_text], "findings": [], "recommendations": []}
    summary["model"] = config.OPENAI_MODEL
    cache.write_text(json.dumps(summary, indent=2))
    return summary
