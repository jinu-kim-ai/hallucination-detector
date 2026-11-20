"""Render a Report for the terminal, as JSON, or as a self-contained HTML page."""
from __future__ import annotations

import html
import json

from .schemas import LABELS, Report

COLORS = {"SUPPORTED": "#2e9e5b", "CONTRADICTED": "#d64545", "FABRICATED": "#b5459e", "UNSUPPORTED": "#d98c1f"}
SEVERITY = {"SUPPORTED": 0, "UNSUPPORTED": 1, "FABRICATED": 2, "CONTRADICTED": 3}
ANSI = {"SUPPORTED": "\033[32m", "CONTRADICTED": "\033[31m", "FABRICATED": "\033[35m", "UNSUPPORTED": "\033[33m"}


def to_json(report: Report, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)


def _headline(report: Report) -> str:
    score, rate = report.reliability_score, report.hallucination_rate
    if score is None:
        return "Reliability score: n/a (nothing to check)"
    return f"Hallucination rate: {rate:.0%}   Reliability score: {score:.1f}/100"


def to_text(report: Report, color: bool = True) -> str:
    lines = []
    for v in report.verdicts:
        tag = f"{ANSI[v.label]}{v.label:<12}\033[0m" if color else f"{v.label:<12}"
        lines.append(f"[{v.claim.id:>2}] {tag} conf={v.confidence:.2f}  {v.claim.text}")
        if v.claim.resolved_text != v.claim.text:
            lines.append(f"      checked as: {v.claim.resolved_text}")
        for c in v.citations[:1]:
            lines.append(f"      evidence [{c.source_id} #{c.sentence_index}] ({c.similarity:.2f}): {c.quote}")
        # the evidence line already shows the best similarity; an uncited claim keeps it as a reason
        shown = bool(v.citations) and v.reasons[:1] and v.reasons[0].startswith("best evidence")
        for reason in v.reasons[1:] if shown else v.reasons:
            lines.append(f"      - {reason}")
    for w in report.warnings:
        lines.append(f"warning: {w}")
    counts = ", ".join(f"{k}={n}" for k, n in report.counts().items())
    lines += ["", f"Claims: {len(report.scored)}  ({counts})", _headline(report)]
    return "\n".join(lines)


# ---------------------------------------------------------------------- HTML
def _highlight(text: str, spans: list[tuple[int, int, str, str]]) -> str:
    """spans: (start, end, css style, tooltip). Later spans in the list win where they overlap."""
    cuts = sorted({0, len(text), *[p for s in spans for p in s[:2] if 0 <= p <= len(text)]})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        covering = [s for s in spans if s[0] <= a and b <= s[1]]
        chunk = html.escape(text[a:b])
        if covering:
            _, _, style, tip = covering[-1]
            chunk = f'<mark style="{style}" title="{html.escape(tip)}">{chunk}</mark>'
        out.append(chunk)
    return "".join(out)


def to_html(report: Report, title: str = "Hallucination report") -> str:
    # Response: each claim's span, colored by label; opacity scales with confidence. Most severe drawn last.
    spans = []
    for v in sorted(report.verdicts, key=lambda v: SEVERITY[v.label]):
        alpha = int(40 + 140 * v.confidence)
        style = f"background:{COLORS[v.label]}{alpha:02x};border-bottom:2px solid {COLORS[v.label]}"
        spans.append((v.claim.start, v.claim.end, style, f"claim {v.claim.id}: {v.label} ({v.confidence:.2f})"))
    response_html = _highlight(report.response, spans)

    # Sources: every cited sentence, tagged with the claims that cite it.
    cited: dict[tuple[str, int, int], list[int]] = {}
    for v in report.verdicts:
        for c in v.citations:
            cited.setdefault((c.source_id, *c.source_span), []).append(v.claim.id)
    source_blocks = []
    for doc in report.sources:
        doc_spans = [(a, b, "background:#3b82f633;border-bottom:2px solid #3b82f6",
                      "cited by claim " + ", ".join(map(str, sorted(set(ids)))))
                     for (sid, a, b), ids in cited.items() if sid == doc.id]
        name = html.escape(doc.title or doc.id)
        source_blocks.append(f'<h3>{name} <code>{html.escape(doc.id)}</code></h3>'
                             f'<div class="text">{_highlight(doc.text, doc_spans)}</div>')

    rows = []
    for v in report.verdicts:
        cites = []
        for c in v.citations:
            nli = (f" &middot; NLI entail {c.nli['entailment']:.2f} / contradict {c.nli['contradiction']:.2f}"
                   if c.nli else "")
            cites.append(f'<div class="cite">{html.escape(c.quote)}<div class="meta">'
                         f'{html.escape(c.source_id)} &middot; sentence {c.sentence_index} &middot; '
                         f'similarity {c.similarity:.2f}{nli}</div></div>')
        checked = (f'<div class="meta">checked as: {html.escape(v.claim.resolved_text)}</div>'
                   if v.claim.resolved_text != v.claim.text else "")
        reasons = "<br>".join(html.escape(r) for r in v.reasons) or "&mdash;"
        rows.append(
            f'<tr><td>{v.claim.id}</td><td>{html.escape(v.claim.text)}{checked}</td>'
            f'<td><span class="pill" style="background:{COLORS[v.label]}">{v.label}</span></td>'
            f'<td><div class="bar"><div style="width:{v.confidence * 100:.0f}%;background:{COLORS[v.label]}"></div></div>'
            f'{v.confidence:.2f}</td><td>{"".join(cites) or "&mdash;"}</td><td>{reasons}</td></tr>')

    score = report.reliability_score
    score_html = "n/a" if score is None else f"{score:.1f}/100"
    rate_html = "n/a" if report.hallucination_rate is None else f"{report.hallucination_rate:.0%}"
    legend = " ".join(f'<span class="pill" style="background:{COLORS[lab]}">{lab}</span>' for lab in LABELS)
    warnings = "".join(f'<p class="warn">Warning: {html.escape(w)}</p>' for w in report.warnings)
    counts = "".join(f"<span>{k.title()}: {n}</span>" for k, n in report.counts().items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1150px;margin:2rem auto;padding:0 1rem;color:#222}}
.text{{line-height:1.8;padding:1rem;border:1px solid #ddd;border-radius:8px;white-space:pre-wrap}}
mark{{color:inherit;padding:1px 2px;border-radius:3px}}
table{{border-collapse:collapse;width:100%;margin-top:1rem;font-size:14px}}
td,th{{border-bottom:1px solid #eee;padding:8px;vertical-align:top;text-align:left}}
.pill{{color:#fff;font-size:12px;padding:2px 8px;border-radius:10px;white-space:nowrap}}
.bar{{width:80px;height:6px;background:#eee;border-radius:3px;margin-bottom:4px}}.bar div{{height:100%;border-radius:3px}}
.meta{{color:#888;font-size:12px}} .cite{{margin-bottom:8px}} .stats span{{margin-right:1.5rem}}
.warn{{background:#fff7e6;border-left:4px solid #d98c1f;padding:6px 10px}} code{{font-size:12px;color:#666}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="stats"><span><b>Reliability score:</b> {score_html}</span>
<span><b>Hallucination rate:</b> {rate_html}</span><span><b>Claims:</b> {len(report.scored)}</span>{counts}</p>
{warnings}<p>{legend}</p>
<h2>Response</h2><div class="text">{response_html}</div>
<h2>Claims</h2>
<table><tr><th>#</th><th>Claim</th><th>Label</th><th>Confidence</th><th>Citations</th><th>Why</th></tr>
{"".join(rows)}</table>
<h2>Sources</h2>{"".join(source_blocks) or "<p>No sources provided.</p>"}
</body></html>"""
