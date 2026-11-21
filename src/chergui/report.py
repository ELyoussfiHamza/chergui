"""Build the single-file HTML report for a run: map, charts, verdicts, timeline, AI summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .analyze import analyze

log = logging.getLogger("chergui.report")

TEMPLATE = Path(__file__).with_name("report_template.html")


def build(run_dir: Path, ai: bool = True) -> Path:
    run_dir = Path(run_dir)
    analysis = analyze(run_dir)
    summary = None
    if ai:
        try:
            from .summary import summarize
            summary = summarize(analysis, run_dir)
            if summary is None:
                log.warning("no OPENAI_API_KEY in .env: report built without the AI summary")
        except Exception as exc:  # the report must never fail because of the AI step
            log.warning("AI summary failed (%s): report built without it", exc)

    data = dict(analysis, summary=summary)
    blob = json.dumps(data, default=float).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", blob)
    out = run_dir / "report.html"
    out.write_text(html)
    return out
