"""Claim-level hallucination detection for LLM answers grounded in a source document."""
from .claims import Claim, ClaimExtractor
from .detector import (
    CONTRADICTED,
    FABRICATED,
    LABELS,
    SUPPORTED,
    UNSUPPORTED,
    ClaimResult,
    DetectionReport,
    HallucinationDetector,
)

__all__ = [
    "Claim", "ClaimExtractor", "HallucinationDetector", "DetectionReport", "ClaimResult",
    "LABELS", "SUPPORTED", "CONTRADICTED", "FABRICATED", "UNSUPPORTED",
]
