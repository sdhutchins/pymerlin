"""Structural checks for the provenance-first benchmark registry."""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
REGISTRY_PATH = REPOSITORY_ROOT / "benchmarks" / "datasets.json"
ALLOWED_STATUSES = {"active", "candidate", "blocked", "excluded"}
ALLOWED_ACCESS = {"bundled", "mixed", "permission_required", "private"}


def test_benchmark_registry_has_unique_classified_datasets() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    datasets = registry["datasets"]
    dataset_ids = [dataset["id"] for dataset in datasets]

    assert registry["schema_version"] == 1
    assert registry["download_by_default"] is False
    assert len(dataset_ids) == len(set(dataset_ids))
    assert {dataset["status"] for dataset in datasets} <= ALLOWED_STATUSES
    assert {dataset["access"] for dataset in datasets} <= ALLOWED_ACCESS


def test_nonbundled_benchmarks_record_provenance_and_activation_work() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())

    for dataset in registry["datasets"]:
        if dataset["access"] == "bundled":
            continue

        assert dataset["paper"]["doi"]
        assert dataset["merlin_evidence"]["description"]
        assert dataset["activation_requirements"]


def test_only_bundled_data_are_active() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    active_datasets = [
        dataset for dataset in registry["datasets"] if dataset["status"] == "active"
    ]

    assert active_datasets
    assert all(dataset["access"] == "bundled" for dataset in active_datasets)
