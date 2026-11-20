"""Hallucination Detector UI:  streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from hallucination_detector import Config, HallucinationDetector
from hallucination_detector.report import to_html, to_json

st.set_page_config(page_title="Hallucination Detector", layout="wide")
st.title("Hallucination Detector")
st.caption("Checks every claim in an AI-generated response against your source documents.")

EXAMPLE = Path(__file__).parent / "examples" / "request.json"


def fill_blank_ids(sources: list[dict]) -> list[dict]:
    """Strip the ids and give blank ones a 'source-N' that no other source uses."""
    taken = {s["id"].strip() for s in sources}
    out = []
    for i, s in enumerate(sources, 1):
        doc_id, n = s["id"].strip(), i
        while not doc_id:
            doc_id = f"source-{n}" if f"source-{n}" not in taken else ""
            n += 1
        taken.add(doc_id)
        out.append({**s, "id": doc_id})
    return out


def input_error(response: str, sources: list[dict]) -> str | None:
    """Why the input cannot be checked, or None. Shown with st.error instead of a traceback."""
    if not response.strip():
        return "Enter a response to check."
    ids = [s["id"] for s in sources]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        return f"Each source needs a different id (used more than once: {', '.join(repeated)})."
    return None


@st.cache_resource(show_spinner="Loading models (first run downloads about 380 MB)...")
def load_detector(use_nli: bool, support_threshold: float) -> HallucinationDetector:
    return HallucinationDetector(Config(use_nli=use_nli, support_threshold=support_threshold))


example = json.loads(EXAMPLE.read_text(encoding="utf-8")) if EXAMPLE.exists() else {"response": "", "sources": []}

with st.sidebar:
    st.header("Settings")
    use_nli = st.toggle("Use the NLI model", value=True,
                        help="Catches meaning changes that keep every name and number. Slower.")
    support = st.slider("Support threshold", 0.45, 0.90, Config().support_threshold, 0.025,
                        help="Similarity at which evidence counts as support on its own.")
    n_sources = st.number_input("Source documents", 1, 8, max(1, len(example["sources"])))

left, right = st.columns(2)
with left:
    st.subheader("Response to check")
    response = st.text_area("Response", example["response"], height=300, label_visibility="collapsed")
with right:
    st.subheader("Source documents")
    sources = []
    for i in range(int(n_sources)):
        default = example["sources"][i] if i < len(example["sources"]) else {"id": f"source-{i + 1}", "text": ""}
        with st.expander(default.get("title") or default["id"], expanded=i == 0):
            doc_id = st.text_input("Id", default["id"], key=f"id{i}")
            text = st.text_area("Text", default["text"], height=140, key=f"text{i}")
        sources.append({"id": doc_id, "title": default.get("title", ""), "text": text})

if st.button("Check response", type="primary"):
    sources = fill_blank_ids(sources)
    error = input_error(response, sources)
    if error:
        st.error(error)
        st.stop()
    try:
        with st.spinner("Checking claims..."):
            report = load_detector(use_nli, support).detect(response, sources)
    except ValueError as e:                         # anything else the library rejects as input
        st.error(f"Could not check this input: {e}")
        st.stop()

    for warning in report.warnings:
        st.warning(warning)
    score, rate, counts = report.reliability_score, report.hallucination_rate, report.counts()
    cols = st.columns(6)
    cols[0].metric("Reliability score", "n/a" if score is None else f"{score:.1f} / 100")
    cols[1].metric("Hallucination rate", "n/a" if rate is None else f"{rate:.0%}")
    for col, (label, n) in zip(cols[2:], counts.items()):
        col.metric(label.title(), n)

    if report.verdicts:
        st.subheader("Claims")
        st.dataframe(pd.DataFrame([{
            "#": v.claim.id, "claim": v.claim.text, "label": v.label, "confidence": v.confidence,
            "cited source": v.citations[0].source_id if v.citations else "",
            "evidence": v.citations[0].quote if v.citations else "",
            "why": "; ".join(v.reasons[1:] or v.reasons),
        } for v in report.verdicts]), hide_index=True, width="stretch")

        st.subheader("Highlighted report")
        page = to_html(report)
        components.html(page, height=900, scrolling=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download JSON", to_json(report), "report.json", "application/json")
        d2.download_button("Download HTML", page, "report.html", "text/html")
