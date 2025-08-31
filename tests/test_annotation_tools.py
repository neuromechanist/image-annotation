"""Tests for annotation manipulation tools."""

import json
import tempfile
from pathlib import Path

import pytest

from image_annotation.tools import (
    batch_remove_model,
    batch_reorder_annotations,
    export_to_csv,
    filter_annotations_by_tokens,
    get_annotation_stats,
    remove_model,
    reorder_annotations,
)


@pytest.fixture
def sample_annotation_data():
    """Create sample annotation data for testing."""
    return {
        "image": {
            "path": "images/test.png",
            "name": "test.png",
            "id": "test001",
        },
        "timestamp": "2025-08-30T15:42:51.595095+00:00",
        "annotations": [
            {
                "model": "model_a",
                "temperature": 0.3,
                "prompts": {
                    "general_description": {
                        "prompt_text": "Describe the image",
                        "response": "Test response A",
                        "token_metrics": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                        },
                        "performance_metrics": {
                            "generation_duration_ms": 1000,
                        },
                    }
                },
            },
            {
                "model": "model_b",
                "temperature": 0.3,
                "prompts": {
                    "general_description": {
                        "prompt_text": "Describe the image",
                        "response": "Test response B",
                        "token_metrics": {
                            "input_tokens": 120,
                            "output_tokens": 80,
                            "total_tokens": 200,
                        },
                        "performance_metrics": {
                            "generation_duration_ms": 1500,
                        },
                    }
                },
            },
            {
                "model": "model_c",
                "temperature": 0.3,
                "prompts": {
                    "general_description": {
                        "prompt_text": "Describe the image",
                        "response": "Test response C",
                        "token_metrics": {
                            "input_tokens": 90,
                            "output_tokens": 60,
                            "total_tokens": 150,
                        },
                        "performance_metrics": {
                            "generation_duration_ms": 1200,
                        },
                    }
                },
            },
        ],
    }


@pytest.fixture
def temp_annotation_file(sample_annotation_data):
    """Create a temporary annotation file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_annotation_data, f, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


def test_reorder_annotations(temp_annotation_file):
    """Test reordering annotations in a file."""
    # Define new order
    new_order = ["model_c", "model_a", "model_b"]

    # Reorder annotations
    result = reorder_annotations(temp_annotation_file, new_order, in_place=True)

    # Verify order
    assert [ann["model"] for ann in result["annotations"]] == new_order

    # Verify file was updated
    with open(temp_annotation_file) as f:
        data = json.load(f)
    assert [ann["model"] for ann in data["annotations"]] == new_order


def test_reorder_annotations_partial_order(temp_annotation_file):
    """Test reordering with partial model list."""
    # Define partial order (only some models)
    partial_order = ["model_b", "model_a"]

    # Reorder annotations
    result = reorder_annotations(temp_annotation_file, partial_order, in_place=True)

    # Verify order - specified models first, then remaining
    models = [ann["model"] for ann in result["annotations"]]
    assert models[0] == "model_b"
    assert models[1] == "model_a"
    assert models[2] == "model_c"  # Not in order list, added at end


def test_remove_model(temp_annotation_file):
    """Test removing a model from annotations."""
    # Remove model_b
    result = remove_model(temp_annotation_file, "model_b", in_place=True)

    # Verify model was removed
    models = [ann["model"] for ann in result["annotations"]]
    assert "model_b" not in models
    assert len(models) == 2
    assert models == ["model_a", "model_c"]

    # Verify file was updated
    with open(temp_annotation_file) as f:
        data = json.load(f)
    assert len(data["annotations"]) == 2


def test_remove_nonexistent_model(temp_annotation_file, capsys):
    """Test removing a model that doesn't exist."""
    # Try to remove non-existent model
    result = remove_model(temp_annotation_file, "model_x", in_place=False)

    # Should not change annotations
    assert len(result["annotations"]) == 3

    # Should print message
    captured = capsys.readouterr()
    assert "Model 'model_x' not found" in captured.out


def test_filter_annotations_by_tokens(temp_annotation_file):
    """Test filtering annotations by token count."""
    # Filter to keep only annotations with < 180 total tokens
    result = filter_annotations_by_tokens(temp_annotation_file, max_tokens=180, in_place=False)

    # Should keep model_a (150 tokens) and model_c (150 tokens)
    models = [ann["model"] for ann in result["annotations"]]
    assert len(models) == 2
    assert "model_a" in models
    assert "model_c" in models
    assert "model_b" not in models  # Has 200 tokens


def test_get_annotation_stats(tmp_path):
    """Test getting statistics from annotation files."""
    # Create multiple test files
    for i in range(3):
        file_data = {
            "image": {"id": f"test{i:03d}"},
            "annotations": [
                {"model": "model_a"},
                {"model": "model_b"},
            ],
        }
        if i == 2:  # Add extra model to last file
            file_data["annotations"].append({"model": "model_c"})

        with open(tmp_path / f"shared{i:04d}.json", "w") as f:
            json.dump(file_data, f)

    # Get statistics
    stats = get_annotation_stats(tmp_path, pattern="shared*.json")

    assert stats["files_processed"] == 3
    assert stats["total_annotations"] == 7  # 2 + 2 + 3
    assert stats["model_counts"]["model_a"] == 3
    assert stats["model_counts"]["model_b"] == 3
    assert stats["model_counts"]["model_c"] == 1
    assert set(stats["models"]) == {"model_a", "model_b", "model_c"}


def test_batch_reorder_annotations(tmp_path, sample_annotation_data):
    """Test batch reordering of annotation files."""
    # Create multiple test files
    for i in range(3):
        with open(tmp_path / f"shared{i:04d}.json", "w") as f:
            json.dump(sample_annotation_data, f)

    # Define new order
    new_order = ["model_c", "model_a", "model_b"]

    # Batch reorder
    count = batch_reorder_annotations(tmp_path, new_order, pattern="shared*.json")

    assert count == 3

    # Verify all files were reordered
    for i in range(3):
        with open(tmp_path / f"shared{i:04d}.json") as f:
            data = json.load(f)
        assert [ann["model"] for ann in data["annotations"]] == new_order


def test_batch_remove_model(tmp_path, sample_annotation_data):
    """Test batch removal of a model from multiple files."""
    # Create multiple test files
    for i in range(3):
        with open(tmp_path / f"shared{i:04d}.json", "w") as f:
            json.dump(sample_annotation_data, f)

    # Batch remove model_b
    count = batch_remove_model(tmp_path, "model_b", pattern="shared*.json")

    assert count == 3

    # Verify model was removed from all files
    for i in range(3):
        with open(tmp_path / f"shared{i:04d}.json") as f:
            data = json.load(f)
        models = [ann["model"] for ann in data["annotations"]]
        assert "model_b" not in models
        assert len(models) == 2


def test_export_to_csv(tmp_path, sample_annotation_data):
    """Test exporting annotations to CSV format."""
    # Create test file
    with open(tmp_path / "shared0001.json", "w") as f:
        json.dump(sample_annotation_data, f)

    # Export to CSV
    csv_file = tmp_path / "annotations.csv"
    export_to_csv(tmp_path, csv_file, pattern="shared*.json", include_metrics=True)

    # Verify CSV was created
    assert csv_file.exists()

    # Read and verify CSV content
    import csv

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3  # One row per model

    # Check first row
    assert rows[0]["image_id"] == "test001"
    assert rows[0]["model"] == "model_a"
    assert rows[0]["prompt_type"] == "general_description"
    assert rows[0]["response"] == "Test response A"
    assert rows[0]["total_tokens"] == "150"
    assert rows[0]["generation_duration_ms"] == "1000"


def test_reorder_with_output_path(temp_annotation_file, tmp_path):
    """Test reordering with output to different file."""
    output_file = tmp_path / "reordered.json"
    new_order = ["model_c", "model_a", "model_b"]

    # Reorder to new file
    reorder_annotations(
        temp_annotation_file, new_order, in_place=False, output_path=output_file
    )

    # Original file should be unchanged
    with open(temp_annotation_file) as f:
        original_data = json.load(f)
    assert [ann["model"] for ann in original_data["annotations"]] == [
        "model_a",
        "model_b",
        "model_c",
    ]

    # New file should have reordered data
    with open(output_file) as f:
        new_data = json.load(f)
    assert [ann["model"] for ann in new_data["annotations"]] == new_order
