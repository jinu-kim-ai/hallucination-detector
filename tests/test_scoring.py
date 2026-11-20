import pytest

from hallucination_detector import Config, Report
from hallucination_detector.schemas import (
    CONTRADICTED, FABRICATED, SUPPORTED, UNSUPPORTED, Citation, Claim, ClaimVerdict,
)
from hallucination_detector.scoring import decide, decisive_nli

CFG = Config()
ENTAILS = {"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04}
CONTRADICTS = {"entailment": 0.01, "contradiction": 0.97, "neutral": 0.02}
NEUTRAL = {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90}


def cite(similarity, nli=None, source_id="a"):
    return Citation(source_id, 0, (0, 1), "quote", similarity, nli)


# ---------------------------------------------------------------- label rules
def test_rule_1_fact_conflict_means_contradicted():
    label, conf, reasons = decide(0.85, ["number mismatch: claim says 45, evidence says 21.5"], [], None, CFG)
    assert label == CONTRADICTED and "number mismatch" in reasons[1]


def test_rule_1_nli_contradiction_needs_close_evidence():
    assert decide(0.80, [], [], CONTRADICTS, CFG)[0] == CONTRADICTED
    assert decide(0.60, [], [], CONTRADICTS, CFG)[0] == UNSUPPORTED      # too far to be the same fact


def test_rule_2_novel_facts_mean_fabricated():
    label, _, reasons = decide(0.40, [], ["300", "'Stanford University'"], None, CFG)
    assert label == FABRICATED and "Stanford University" in reasons[-1]


def test_rule_order_contradiction_beats_fabrication():
    assert decide(0.90, ["number mismatch"], ["1938"], None, CFG)[0] == CONTRADICTED


def test_rule_3_similarity_alone_can_support():
    assert decide(0.65, [], [], None, CFG)[0] == SUPPORTED
    assert decide(0.64, [], [], None, CFG)[0] == UNSUPPORTED


def test_rule_3_entailment_supports_a_looser_paraphrase():
    label, _, reasons = decide(0.50, [], [], ENTAILS, CFG)
    assert label == SUPPORTED and any("NLI entailment" in r for r in reasons)
    assert decide(0.40, [], [], ENTAILS, CFG)[0] == UNSUPPORTED          # unrelated text cannot entail


def test_rule_4_otherwise_unsupported():
    assert decide(0.20, [], [], NEUTRAL, CFG)[0] == UNSUPPORTED


@pytest.mark.parametrize("sim", [-0.2, 0.0, 0.3, 0.5, 0.65, 0.8, 1.0])
@pytest.mark.parametrize("conflicts, novel, nli", [
    ([], [], None), (["a", "b", "c", "d"], [], None), ([], ["x", "y", "z", "w", "v"], None),
    ([], [], ENTAILS), ([], [], CONTRADICTS),
])
def test_confidence_stays_in_range(sim, conflicts, novel, nli):
    _, conf, _ = decide(sim, conflicts, novel, nli, CFG)
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------- which NLI result counts
def test_contradiction_only_counts_from_the_best_sentence():
    # a neighbor about a different fact contradicts; the best sentence is merely neutral
    assert decisive_nli([cite(0.90, NEUTRAL), cite(0.75, CONTRADICTS)], CFG) == NEUTRAL
    assert decisive_nli([cite(0.90, CONTRADICTS), cite(0.75, ENTAILS)], CFG) == CONTRADICTS


def test_entailment_may_come_from_any_related_sentence():
    assert decisive_nli([cite(0.80, NEUTRAL), cite(0.60, ENTAILS)], CFG) == ENTAILS
    assert decisive_nli([cite(0.80, NEUTRAL), cite(0.30, ENTAILS)], CFG) == NEUTRAL     # not related


def test_no_nli_scores():
    assert decisive_nli([cite(0.9), cite(0.8)], CFG) is None


# ---------------------------------------------------------------- response-level score
def verdict(i, label, confidence, duplicate_of=None):
    return ClaimVerdict(Claim(i, f"claim {i}.", f"claim {i}.", 0, 1, duplicate_of=duplicate_of), label, confidence)


def test_reliability_score_is_confidence_weighted():
    report = Report("r", [verdict(1, SUPPORTED, 0.9), verdict(2, CONTRADICTED, 0.6)])
    assert report.reliability_score == pytest.approx(100 * 0.9 / 1.5)
    assert report.hallucination_rate == 0.5
    assert report.counts() == {SUPPORTED: 1, CONTRADICTED: 1, FABRICATED: 0, UNSUPPORTED: 0}


def test_duplicates_count_once():
    report = Report("r", [verdict(1, SUPPORTED, 0.8), verdict(2, FABRICATED, 0.8), verdict(3, FABRICATED, 0.8, 2)])
    assert report.reliability_score == pytest.approx(50.0)
    assert report.hallucination_rate == 0.5


def test_no_claims_means_no_score():
    report = Report("r", [])
    assert report.reliability_score is None and report.hallucination_rate is None
    assert report.to_dict()["reliability_score"] is None
