"""Render detection results for the terminal and as a highlighted HTML report."""
from __future__ import annotations

import html

from .detector import DetectionReport, LABELS

COLORS = {
    "SUPPORTED": "#2e9e5b",
    "CONTRADICTED": "#d64545",
    "FABRICATED": "#b5459e",
    "UNSUPPORTED": "#d98c1f",
}
SEVERITY = {"SUPPORTED": 0, "UNSUPPORTED": 1, "FABRICATED": 2, "CONTRADICTED": 3}
ANSI = {"SUPPORTED": "\033[32m", "CONTRADICTED": "\033[31m", "FABRICATED": "\033[35m", "UNSUPPORTED": "\033[33m"}


def to_text(report: DetectionReport, color: bool = True) -> str:
    lines = []
    for i, r in enumerate(report.results, 1):
        tag = f"{ANSI[r.label]}{r.label:<12}\033[0m" if color else f"{r.label:<12}"
        lines.append(f"[{i:>2}] {tag} conf={r.confidence:.2f}  {r.claim}")
        if r.evidence:
            lines.append(f"      evidence ({r.evidence[0].similarity:.2f}): {r.evidence[0].sentence}")
        for reason in r.reasons[1:]:
            lines.append(f"      - {reason}")
    counts = ", ".join(f"{k}={v}" for k, v in report.counts().items())
    lines.append("")
    lines.append(f"Claims: {len(report.results)}  ({counts})")
    lines.append(f"Hallucination rate: {report.hallucination_rate:.0%}   Reliability score: {report.reliability_score:.1f}/100")
    return "\n".join(lines)


def to_html(report: DetectionReport, title: str = "Hallucination report") -> str:
    # Highlight each answer sentence by the most severe label among its claims;
    # opacity of the highlight scales with confidence.
    worst: dict[str, tuple[str, float]] = {}
    for r in report.results:
        cur = worst.get(r.sentence)
        if cur is None or SEVERITY[r.label] > SEVERITY[cur[0]]:
            worst[r.sentence] = (r.label, r.confidence)

    answer_html = html.escape(report.answer)
    for sentence, (label, conf) in worst.items():
        alpha = int(40 + 140 * conf)
        span = (
            f'<mark style="background:{COLORS[label]}{alpha:02x};border-bottom:2px solid {COLORS[label]}" '
            f'title="{label} ({conf:.2f})">{html.escape(sentence)}</mark>'
        )
        answer_html = answer_html.replace(html.escape(sentence), span, 1)

    rows = []
    for i, r in enumerate(report.results, 1):
        ev = html.escape(r.evidence[0].sentence) if r.evidence else "&mdash;"
        sim = f"{r.evidence[0].similarity:.2f}" if r.evidence else "&mdash;"
        reasons = "<br>".join(html.escape(x) for x in r.reasons[1:]) or "&mdash;"
        rows.append(
            f"<tr><td>{i}</td><td>{html.escape(r.claim)}</td>"
            f'<td><span class="pill" style="background:{COLORS[r.label]}">{r.label}</span></td>'
            f'<td><div class="bar"><div style="width:{r.confidence*100:.0f}%;background:{COLORS[r.label]}"></div></div>'
            f"{r.confidence:.2f}</td><td>{ev}<div class='sim'>similarity {sim}</div></td><td>{reasons}</td></tr>"
        )

    legend = " ".join(f'<span class="pill" style="background:{COLORS[l]}">{l}</span>' for l in LABELS)
    counts = report.counts()
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
.answer{{line-height:1.8;padding:1rem;border:1px solid #ddd;border-radius:8px;white-space:pre-wrap}}
mark{{color:inherit;padding:1px 2px;border-radius:3px}}
table{{border-collapse:collapse;width:100%;margin-top:1rem;font-size:14px}}
td,th{{border-bottom:1px solid #eee;padding:8px;vertical-align:top;text-align:left}}
.pill{{color:#fff;font-size:12px;padding:2px 8px;border-radius:10px;white-space:nowrap}}
.bar{{width:80px;height:6px;background:#eee;border-radius:3px;margin-bottom:4px}}.bar div{{height:100%;border-radius:3px}}
.sim{{color:#888;font-size:12px}} .stats span{{margin-right:1.5rem}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="stats"><span><b>Reliability score:</b> {report.reliability_score:.1f}/100</span>
<span><b>Hallucination rate:</b> {report.hallucination_rate:.0%}</span>
<span><b>Claims:</b> {len(report.results)}</span>
{"".join(f"<span>{k.title()}: {v}</span>" for k, v in counts.items())}</p>
<p>{legend}</p>
<h2>Answer</h2><div class="answer">{answer_html}</div>
<h2>Claims</h2>
<table><tr><th>#</th><th>Claim</th><th>Label</th><th>Confidence</th><th>Best evidence</th><th>Why</th></tr>
{"".join(rows)}</table></body></html>"""
