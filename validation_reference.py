"""The immutable course holdout, shared by preparation and training preflight."""
from __future__ import annotations

import hashlib
from pathlib import Path

VALIDATION_SHA256 = "cd9b0e5cbf20e45922905d249a1dfe397d66abe886aa622d2632eaeb604e81a8"
TOKENIZER_SHA256 = "78025606dd6a505cd74f4b187866bf5540ee83239131aa2f1ddefdfd2e104272"
VALIDATION_TOKENS = 5_000_001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_validation_reference(path: Path, expected_tokens: int = VALIDATION_TOKENS) -> str:
    if expected_tokens != VALIDATION_TOKENS or path.stat().st_size != VALIDATION_TOKENS * 2:
        raise ValueError(f"Frozen course validation requires {VALIDATION_TOKENS:,} tokens: {path}")
    actual = sha256_file(path)
    if actual != VALIDATION_SHA256:
        raise ValueError(
            f"Validation SHA256 mismatch for {path}: expected {VALIDATION_SHA256}, got {actual}. "
            "Use the bundled reference and prepare into a fresh directory; do not replace "
            "validation in an existing corpus or edit its hash."
        )
    return actual


def verify_tokenizer_reference(path: Path) -> str:
    actual = sha256_file(path)
    if actual != TOKENIZER_SHA256:
        raise ValueError(
            f"Tokenizer SHA256 mismatch for {path}: expected {TOKENIZER_SHA256}, got {actual}"
        )
    return actual
