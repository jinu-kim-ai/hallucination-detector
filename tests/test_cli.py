import io
import json
import sys
from pathlib import Path

import pytest

from hallucination_detector import Config
from hallucination_detector.cli import build_parser, config_from_args, main

ROOT = Path(__file__).parent.parent
CLAIM = "Marie Curie was born in Warsaw in 1867."


def _request(tmp_path: Path, data, name: str = "request.json") -> Path:
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def _one_line_error(capsys) -> str:
    err = capsys.readouterr().err
    assert err.startswith("hallucheck: error: ") and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    return err


# ---------------------------------------------------------------------- 13: UTF-8 BOM input
def test_answer_file_with_a_bom_is_read_without_it(tmp_path, capsys):
    (tmp_path / "answer.txt").write_bytes(b"\xef\xbb\xbf" + CLAIM.encode())
    (tmp_path / "source.txt").write_text(CLAIM, encoding="utf-8")
    assert main(["--no-nli", "--embedder", "hashing", "detect", "--answer", str(tmp_path / "answer.txt"),
                 "--source", str(tmp_path / "source.txt"), "--json", "-"]) == 0
    claims = json.loads(capsys.readouterr().out)["claims"]
    assert claims[0]["text"] == CLAIM


# ---------------------------------------------------------------------- 14: piped stdout on Windows
def test_non_cp1252_text_prints_when_stdout_is_a_cp1252_pipe(tmp_path, monkeypatch):
    request = _request(tmp_path, {"response": "Marie Skłodowska Curie was born in Warsaw in 1867.",
                                  "sources": [{"id": "curie", "text": "Maria Skłodowska Curie was born in Warsaw."}]})
    for args in (["--json", "-"], []):
        buffer = io.BytesIO()
        # What a redirected stdout looks like on Windows without PYTHONIOENCODING / PYTHONUTF8.
        monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252"))
        assert main(["--no-nli", "--embedder", "hashing", "detect", "--request", str(request), *args]) == 0
        sys.stdout.flush()
        assert "Skłodowska" in buffer.getvalue().decode("utf-8")


# ---------------------------------------------------------------------- 32: every threshold on the CLI
def test_cli_defaults_match_config():
    assert config_from_args(build_parser().parse_args(["evaluate"])) == Config()
    assert config_from_args(build_parser().parse_args(["--no-nli", "evaluate"])) == Config(use_nli=False)


def test_cli_overrides_every_threshold():
    args = build_parser().parse_args([
        "--support-threshold", "0.6", "--related-threshold", "0.4", "--contradiction-threshold", "0.8",
        "--nli-threshold", "0.9", "--top-k", "5", "--nli-top-k", "2", "--nli-context-sentences", "1",
        "--min-claim-words", "4", "evaluate"])
    assert config_from_args(args) == Config(support_threshold=0.6, related_threshold=0.4,
                                            contradiction_threshold=0.8, nli_threshold=0.9, top_k=5,
                                            nli_top_k=2, nli_context_sentences=1, min_claim_words=4)


@pytest.mark.parametrize("flag, value", [("--support-threshold", "1.5"), ("--nli-threshold", "-0.1"),
                                         ("--related-threshold", "abc"), ("--top-k", "0"),
                                         ("--nli-top-k", "0"), ("--nli-context-sentences", "-1"),
                                         ("--min-claim-words", "0")])
def test_cli_rejects_out_of_range_numbers(flag, value, capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([flag, value, "evaluate"])
    assert exit_info.value.code == 2
    assert f"argument {flag}" in capsys.readouterr().err


# ---------------------------------------------------------------------- 34: one-line errors, exit 2
def test_missing_request_file_is_a_one_line_error(tmp_path, capsys):
    assert main(["--no-nli", "detect", "--request", str(tmp_path / "nope.json")]) == 2
    assert "nope.json" in _one_line_error(capsys)


@pytest.mark.parametrize("content, expected", [
    ("{not json", "request.json"),
    ({"response": CLAIM, "sources": [{"id": "a", "text": CLAIM}, {"id": "a", "text": CLAIM}]}, "unique"),
    ({"response": 42, "sources": [CLAIM]}, "'response' must be a string"),
    ([CLAIM], "request.json"),
])
def test_bad_request_is_a_one_line_error(tmp_path, capsys, content, expected):
    assert main(["--no-nli", "detect", "--request", str(_request(tmp_path, content))]) == 2
    assert expected in _one_line_error(capsys)


def test_source_that_is_not_utf8_is_a_one_line_error(tmp_path, capsys):
    (tmp_path / "answer.txt").write_text(CLAIM, encoding="utf-8")
    (tmp_path / "latin1.txt").write_bytes("Marie Curie était née à Varsovie.".encode("latin-1"))
    assert main(["--no-nli", "detect", "--answer", str(tmp_path / "answer.txt"),
                 "--source", str(tmp_path / "latin1.txt")]) == 2
    assert "latin1.txt" in _one_line_error(capsys)


def test_missing_dataset_is_a_one_line_error(tmp_path, capsys):
    assert main(["--no-nli", "evaluate", "--data", str(tmp_path / "no-such-dir")]) == 2
    assert "sources.json" in _one_line_error(capsys)


def test_detect_creates_output_directories(tmp_path, capsys):
    request = _request(tmp_path, {"response": CLAIM, "sources": [CLAIM]})
    html_path, json_path = tmp_path / "out" / "x.html", tmp_path / "out" / "deeper" / "x.json"
    assert main(["--no-nli", "--embedder", "hashing", "detect", "--request", str(request),
                 "--html", str(html_path), "--json", str(json_path)]) == 0
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert json.loads(json_path.read_text(encoding="utf-8"))["claims"]


def test_unwritable_output_is_a_one_line_error_before_the_report(tmp_path, capsys):
    request = _request(tmp_path, {"response": CLAIM, "sources": [CLAIM]})
    (tmp_path / "file").write_text("", encoding="utf-8")       # a file where a directory is needed
    assert main(["--no-nli", "--embedder", "hashing", "detect", "--request", str(request),
                 "--html", str(tmp_path / "file" / "x.html")]) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("hallucheck: error: ") and "Traceback" not in err


# ---------------------------------------------------------------------- 27: Streamlit input checks
@pytest.fixture
def app():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    at.toggle[0].set_value(False)                   # no NLI: keeps the one full run fast
    return at


@pytest.mark.parametrize("ids, response, expected", [
    (("x", "x"), None, "different id"),
    (("x ", " x"), None, "different id"),
    (("mission-summary", "crew-and-return"), "   ", "Enter a response"),
])
def test_app_reports_invalid_input_with_st_error(app, ids, response, expected):
    app.text_input(key="id0").set_value(ids[0])
    app.text_input(key="id1").set_value(ids[1])
    if response is not None:
        app.text_area[0].set_value(response)
    app.button[0].click().run()
    assert not app.exception
    assert any(expected in e.value for e in app.error)


def test_app_gives_blank_ids_a_unique_default(app):
    app.text_input(key="id0").set_value("source-2")
    app.text_input(key="id1").set_value("")         # used to become 'source-2' too and crash in as_documents
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].label == "Reliability score"
