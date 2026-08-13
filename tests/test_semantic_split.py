"""Tests for Semantic Stratified Splitting in `soup data split`."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.data import _get_semantic_labels

runner = CliRunner()


@pytest.fixture
def dummy_dataset(tmp_path):
    ds_path = tmp_path / "dataset.jsonl"
    rows = [
        {"text": "Python programming language and coding", "category": "code"},
        {"text": "Writing code in python", "category": "code"},
        {"text": "Python software development", "category": "code"},
        {"text": "Grade school basic math and arithmetic", "category": "math"},
        {"text": "Solving equations and numbers", "category": "math"},
        {"text": "Basic algebra mathematics tutor", "category": "math"},
        {"text": "Creative storytelling writing poetry", "category": "writing"},
        {"text": "Novel author fantasy book prompt", "category": "writing"},
        {"text": "Short story narrative creative writer", "category": "writing"},
        {"text": "How to write screenplays and drama", "category": "writing"},
    ]
    ds_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return ds_path


class TestSemanticSplit:
    def test_stratify_exclusivity(self, dummy_dataset):
        """Cannot use --stratify and --stratify-semantic together."""
        result = runner.invoke(
            app,
            [
                "data", "split", str(dummy_dataset),
                "--val", "20",
                "--stratify", "category",
                "--stratify-semantic",
            ],
        )
        assert result.exit_code == 1
        assert "Cannot use --stratify and --stratify-semantic together" in result.output

    def test_invalid_num_clusters(self, dummy_dataset):
        """--num-clusters must be a positive integer."""
        result = runner.invoke(
            app,
            [
                "data", "split", str(dummy_dataset),
                "--val", "20",
                "--stratify-semantic",
                "--num-clusters", "0",
            ],
        )
        assert result.exit_code == 1
        assert "--num-clusters must be a positive integer" in result.output

    def test_semantic_split_happy_path(self, dummy_dataset, tmp_path, monkeypatch):
        """Happy path for semantic stratified splitting."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "data", "split", str(dummy_dataset),
                "--val", "30",  # absolute count 3
                "--test", "20",  # absolute count 2
                "--stratify-semantic",
                "--num-clusters", "3",
                "--seed", "42",
            ],
        )
        assert result.exit_code == 0

        train_file = tmp_path / "dataset_train.jsonl"
        val_file = tmp_path / "dataset_val.jsonl"
        test_file = tmp_path / "dataset_test.jsonl"

        assert train_file.exists()
        assert val_file.exists()
        assert test_file.exists()

        train_rows = [
            json.loads(line)
            for line in train_file.read_text(encoding="utf-8").splitlines()
        ]
        val_rows = [
            json.loads(line) for line in val_file.read_text(encoding="utf-8").splitlines()
        ]
        test_rows = [
            json.loads(line)
            for line in test_file.read_text(encoding="utf-8").splitlines()
        ]

        # Verify that all rows are accounted for and no split is empty
        assert len(train_rows) + len(val_rows) + len(test_rows) == 10
        assert len(train_rows) > 0
        assert len(val_rows) > 0
        assert len(test_rows) > 0

    def test_get_semantic_labels_fallback(self, monkeypatch):
        """Test fallback to length bucketing when sklearn is missing."""
        import sys
        # Hide sklearn to force ImportError
        monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
        monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)

        data = [
            {"text": "Short text"},
            {"text": "Medium length text here"},
            {"text": "Very long text that spans across multiple words and sentences for bucketing"},
        ]

        labels = _get_semantic_labels(data, num_clusters=3)
        assert len(labels) == 3
        # Should return bucket-based labels
        assert all(label.startswith("bucket_") for label in labels)

        # Test edge case with single element / single cluster
        labels_single = _get_semantic_labels(data, num_clusters=1)
        assert labels_single == ["bucket_0", "bucket_0", "bucket_0"]
