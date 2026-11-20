"""Claim-level hallucination detection for LLM responses grounded in source documents."""
from .claims import ClaimExtractor
from .config import Config
from .pipeline import HallucinationDetector
from .schemas import (
    CONTRADICTED,
    FABRICATED,
    LABELS,
    SUPPORTED,
    UNSUPPORTED,
    Citation,
    Claim,
    ClaimVerdict,
    Report,
    SourceDocument,
)

__version__ = "2.0.0"
__all__ = [
    "HallucinationDetector", "Config", "ClaimExtractor", "SourceDocument", "Claim", "Citation",
    "ClaimVerdict", "Report", "LABELS", "SUPPORTED", "CONTRADICTED", "FABRICATED", "UNSUPPORTED",
]
