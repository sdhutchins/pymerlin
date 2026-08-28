"""Deterministic process-count validation shared by compute entry points."""

from __future__ import annotations


def validate_workers(workers: int) -> int:
    """Require an explicit positive number of worker processes."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    return workers
