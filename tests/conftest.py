"""Shared fixtures. Models are loaded once per test session."""
import pytest

from hallucination_detector import HallucinationDetector
from hallucination_detector.claims import load_nlp

APOLLO = (
    "Apollo 11 launched from Kennedy Space Center on July 16, 1969. "
    "On July 20, 1969, the lunar module Eagle landed in the Sea of Tranquility. "
    "Armstrong became the first person to walk on the Moon, followed by Aldrin about 19 minutes later. "
    "The astronauts collected about 21.5 kilograms of lunar samples. "
    "The mission returned to Earth on July 24, 1969, splashing down in the Pacific Ocean."
)


@pytest.fixture(scope="session")
def nlp():
    return load_nlp()


@pytest.fixture(scope="session")
def detector(nlp):
    """Default configuration: embeddings + fact checks + NLI."""
    return HallucinationDetector(nlp=nlp)


@pytest.fixture(scope="session")
def detector_no_nli(nlp):
    return HallucinationDetector(nlp=nlp, use_nli=False)


@pytest.fixture(scope="session")
def apollo():
    return APOLLO
