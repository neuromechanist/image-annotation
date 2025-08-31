"""Tools for manipulating annotation JSON files."""

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def reorder_annotations(
    file_path: str | Path,
    model_order: list[str],
    in_place: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Reorder annotations in a JSON file according to specified model order.

    Args:
        file_path: Path to the annotation JSON file
        model_order: List of model names in desired order
        in_place: If True, modify the file in place. If False, return modified data
        output_path: Optional path to save the reordered file (if not in_place)

    Returns:
        The reordered annotation data
    """
    file_path = Path(file_path)

    with open(file_path) as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError(f"No 'annotations' field found in {file_path}")

    # Create a mapping of model names to annotations
    model_map = {ann["model"]: ann for ann in data["annotations"]}

    # Build reordered list
    reordered = []
    for model in model_order:
        if model in model_map:
            reordered.append(model_map[model])

    # Add any remaining models not in the specified order
    for ann in data["annotations"]:
        if ann["model"] not in model_order:
            reordered.append(ann)

    data["annotations"] = reordered

    # Save the file
    if in_place:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    elif output_path:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    return data


def remove_model(
    file_path: str | Path,
    model_name: str,
    in_place: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Remove a specific model's annotations from a JSON file.

    Args:
        file_path: Path to the annotation JSON file
        model_name: Name of the model to remove
        in_place: If True, modify the file in place. If False, return modified data
        output_path: Optional path to save the modified file (if not in_place)

    Returns:
        The modified annotation data
    """
    file_path = Path(file_path)

    with open(file_path) as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError(f"No 'annotations' field found in {file_path}")

    # Filter out the specified model
    original_count = len(data["annotations"])
    data["annotations"] = [ann for ann in data["annotations"] if ann["model"] != model_name]
    removed_count = original_count - len(data["annotations"])

    if removed_count == 0:
        print(f"Model '{model_name}' not found in {file_path}")

    # Save the file
    if in_place:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    elif output_path:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    return data


def batch_reorder_annotations(
    directory: str | Path,
    model_order: list[str],
    pattern: str = "*.json",
    exclude_pattern: str | None = None,
) -> int:
    """
    Reorder annotations in multiple JSON files.

    Args:
        directory: Directory containing annotation files
        model_order: List of model names in desired order
        pattern: Glob pattern to match files (default: "*.json")
        exclude_pattern: Optional pattern to exclude files

    Returns:
        Number of files processed
    """
    directory = Path(directory)
    files = list(directory.glob(pattern))

    if exclude_pattern:
        files = [f for f in files if not f.name.startswith(exclude_pattern)]

    processed = 0
    for file_path in tqdm(files, desc="Reordering annotations"):
        try:
            reorder_annotations(file_path, model_order, in_place=True)
            processed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return processed


def batch_remove_model(
    directory: str | Path,
    model_name: str,
    pattern: str = "*.json",
    exclude_pattern: str | None = None,
) -> int:
    """
    Remove a model's annotations from multiple JSON files.

    Args:
        directory: Directory containing annotation files
        model_name: Name of the model to remove
        pattern: Glob pattern to match files (default: "*.json")
        exclude_pattern: Optional pattern to exclude files

    Returns:
        Number of files processed
    """
    directory = Path(directory)
    files = list(directory.glob(pattern))

    if exclude_pattern:
        files = [f for f in files if not f.name.startswith(exclude_pattern)]

    processed = 0
    removed_total = 0
    for file_path in tqdm(files, desc=f"Removing {model_name}"):
        try:
            with open(file_path) as f:
                data = json.load(f)

            if "annotations" in data:
                original_count = len(data["annotations"])
                remove_model(file_path, model_name, in_place=True)

                # Check how many were removed
                with open(file_path) as f:
                    new_data = json.load(f)
                removed = original_count - len(new_data["annotations"])
                removed_total += removed

                if removed > 0:
                    processed += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Removed {model_name} from {processed} files (total {removed_total} annotations)")
    return processed


def get_annotation_stats(
    directory: str | Path, pattern: str = "shared*.json"
) -> dict[str, Any]:
    """
    Get statistics about annotations in a directory.

    Args:
        directory: Directory containing annotation files
        pattern: Glob pattern to match files

    Returns:
        Dictionary with statistics
    """
    directory = Path(directory)
    files = list(directory.glob(pattern))

    model_counts = {}
    total_annotations = 0
    files_processed = 0

    for file_path in files:
        try:
            with open(file_path) as f:
                data = json.load(f)

            if "annotations" in data:
                files_processed += 1
                for ann in data["annotations"]:
                    model = ann.get("model", "unknown")
                    model_counts[model] = model_counts.get(model, 0) + 1
                    total_annotations += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return {
        "files_processed": files_processed,
        "total_annotations": total_annotations,
        "model_counts": model_counts,
        "models": list(model_counts.keys()),
    }


def filter_annotations_by_tokens(
    file_path: str | Path,
    max_tokens: int | None = None,
    min_tokens: int | None = None,
    in_place: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Filter annotations based on token count criteria.

    Args:
        file_path: Path to the annotation JSON file
        max_tokens: Maximum total tokens allowed
        min_tokens: Minimum total tokens required
        in_place: If True, modify the file in place
        output_path: Optional path to save the filtered file

    Returns:
        The filtered annotation data
    """
    file_path = Path(file_path)

    with open(file_path) as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError(f"No 'annotations' field found in {file_path}")

    filtered_annotations = []

    for ann in data["annotations"]:
        # Calculate total tokens across all prompts
        total_tokens = 0
        for _prompt_key, prompt_data in ann.get("prompts", {}).items():
            if "token_metrics" in prompt_data:
                total_tokens += prompt_data["token_metrics"].get("total_tokens", 0)

        # Apply filters
        if max_tokens and total_tokens > max_tokens:
            continue
        if min_tokens and total_tokens < min_tokens:
            continue

        filtered_annotations.append(ann)

    data["annotations"] = filtered_annotations

    # Save the file
    if in_place:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    elif output_path:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    return data


def export_to_csv(
    directory: str | Path,
    output_file: str | Path,
    pattern: str = "shared*.json",
    include_metrics: bool = False,
) -> None:
    """
    Export annotations to CSV format.

    Args:
        directory: Directory containing annotation files
        output_file: Path to output CSV file
        pattern: Glob pattern to match files
        include_metrics: Whether to include token and performance metrics
    """
    import csv

    directory = Path(directory)
    output_file = Path(output_file)
    files = sorted(directory.glob(pattern))

    # Prepare CSV headers
    headers = ["image_id", "image_path", "model", "prompt_type", "response"]
    if include_metrics:
        headers.extend(["input_tokens", "output_tokens", "total_tokens", "generation_duration_ms"])

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for file_path in tqdm(files, desc="Exporting to CSV"):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                image_id = data["image"]["id"]
                image_path = data["image"]["path"]

                for ann in data.get("annotations", []):
                    model = ann["model"]

                    for prompt_key, prompt_data in ann.get("prompts", {}).items():
                        row = {
                            "image_id": image_id,
                            "image_path": image_path,
                            "model": model,
                            "prompt_type": prompt_key,
                            "response": prompt_data.get("response", ""),
                        }

                        if include_metrics:
                            token_metrics = prompt_data.get("token_metrics", {})
                            perf_metrics = prompt_data.get("performance_metrics", {})
                            row.update(
                                {
                                    "input_tokens": token_metrics.get("input_tokens", ""),
                                    "output_tokens": token_metrics.get("output_tokens", ""),
                                    "total_tokens": token_metrics.get("total_tokens", ""),
                                    "generation_duration_ms": perf_metrics.get(
                                        "generation_duration_ms", ""
                                    ),
                                }
                            )

                        writer.writerow(row)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Exported annotations to {output_file}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python annotation_tools.py <command> [args]")
        print("Commands:")
        print("  stats <directory> - Get annotation statistics")
        print("  reorder <directory> - Reorder annotations to standard order")
        print("  remove <directory> <model> - Remove a model from all files")
        sys.exit(1)

    command = sys.argv[1]

    if command == "stats":
        if len(sys.argv) < 3:
            print("Usage: python annotation_tools.py stats <directory>")
            sys.exit(1)
        stats = get_annotation_stats(sys.argv[2])
        print(f"Files processed: {stats['files_processed']}")
        print(f"Total annotations: {stats['total_annotations']}")
        print("Model counts:")
        for model, count in sorted(stats["model_counts"].items()):
            print(f"  {model}: {count}")

    elif command == "reorder":
        if len(sys.argv) < 3:
            print("Usage: python annotation_tools.py reorder <directory>")
            sys.exit(1)
        model_order = [
            "qwen2.5vl:7b",
            "qwen2.5vl:32b",
            "gemma3:4b",
            "gemma3:12b",
            "gemma3:27b",
            "mistral-small3.2:24b",
        ]
        count = batch_reorder_annotations(sys.argv[2], model_order, pattern="shared*.json")
        print(f"Reordered {count} files")

    elif command == "remove":
        if len(sys.argv) < 4:
            print("Usage: python annotation_tools.py remove <directory> <model>")
            sys.exit(1)
        count = batch_remove_model(sys.argv[2], sys.argv[3], pattern="shared*.json")
        print(f"Processed {count} files")
