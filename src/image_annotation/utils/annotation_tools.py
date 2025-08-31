"""Tools for manipulating annotation JSON files."""

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _normalize_paths(
    paths: str | Path | list[str | Path],
    pattern: str = "*.json",
    exclude_pattern: str | None = None,
) -> list[Path]:
    """Normalize input paths to a list of file paths."""
    if isinstance(paths, list):
        return [Path(p) for p in paths]

    path = Path(paths)
    if path.is_file():
        return [path]
    elif path.is_dir():
        files = list(path.glob(pattern))
        if exclude_pattern:
            files = [f for f in files if not f.name.startswith(exclude_pattern)]
        return files
    else:
        raise ValueError(f"Path does not exist: {path}")


def reorder_annotations(  # noqa: C901
    paths: str | Path | list[str | Path],
    model_order: list[str],
    pattern: str = "*.json",
    exclude_pattern: str | None = None,
    in_place: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | int:
    """
    Reorder annotations in JSON file(s) according to specified model order.

    Args:
        paths: Path to file, directory, or list of paths
        model_order: List of model names in desired order
        pattern: Glob pattern for directory search (default: "*.json")
        exclude_pattern: Optional pattern to exclude files
        in_place: If True, modify files in place
        output_dir: Optional directory to save reordered files

    Returns:
        For single file: The reordered annotation data dict
        For multiple files: Number of files processed
    """
    # Normalize input to list of files
    files = _normalize_paths(paths, pattern, exclude_pattern)

    # Process files
    single_file = len(files) == 1
    processed = 0
    last_data = None

    for file_path in tqdm(files, desc="Reordering annotations", disable=single_file):
        try:
            with open(file_path) as f:
                data = json.load(f)

            if "annotations" not in data:
                if single_file:
                    raise ValueError(f"No 'annotations' field found in {file_path}")
                print(f"Warning: No 'annotations' field in {file_path}")
                continue

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
            last_data = data

            # Save the file
            if in_place:
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
            elif output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / file_path.name
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

            processed += 1

        except Exception as e:
            if single_file:
                raise
            print(f"Error processing {file_path}: {e}")

    return last_data if single_file else processed


def remove_model(  # noqa: C901
    paths: str | Path | list[str | Path],
    model_name: str,
    pattern: str = "*.json",
    exclude_pattern: str | None = None,
    in_place: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | int:
    """
    Remove a specific model's annotations from JSON file(s).

    Args:
        paths: Path to file, directory, or list of paths
        model_name: Name of the model to remove
        pattern: Glob pattern for directory search (default: "*.json")
        exclude_pattern: Optional pattern to exclude files
        in_place: If True, modify files in place
        output_dir: Optional directory to save modified files

    Returns:
        For single file: The modified annotation data dict
        For multiple files: Number of files where model was removed
    """
    # Normalize input to list of files
    files = _normalize_paths(paths, pattern, exclude_pattern)

    # Process files
    single_file = len(files) == 1
    files_with_removals = 0
    total_removed = 0
    last_data = None

    for file_path in tqdm(files, desc=f"Removing {model_name}", disable=single_file):
        try:
            with open(file_path) as f:
                data = json.load(f)

            if "annotations" not in data:
                if single_file:
                    raise ValueError(f"No 'annotations' field found in {file_path}")
                continue

            # Filter out the specified model
            original_count = len(data["annotations"])
            data["annotations"] = [ann for ann in data["annotations"] if ann["model"] != model_name]
            removed_count = original_count - len(data["annotations"])

            if removed_count == 0 and single_file:
                print(f"Model '{model_name}' not found in {file_path}")
            elif removed_count == 0 and not single_file:
                # Silent for batch operations
                continue

            if removed_count > 0:
                files_with_removals += 1
                total_removed += removed_count

            last_data = data

            # Save the file
            if in_place:
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
            elif output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / file_path.name
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            if single_file:
                raise
            print(f"Error processing {file_path}: {e}")

    if not single_file:
        print(
            f"Removed {model_name} from {files_with_removals} files "
            f"(total {total_removed} annotations)"
        )

    return last_data if single_file else files_with_removals


def get_annotation_stats(directory: str | Path, pattern: str = "shared*.json") -> dict[str, Any]:
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


def filter_annotations_by_tokens(  # noqa: C901
    paths: str | Path | list[str | Path],
    max_tokens: int | None = None,
    min_tokens: int | None = None,
    pattern: str = "*.json",
    exclude_pattern: str | None = None,
    in_place: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any] | int:
    """
    Filter annotations based on token count criteria in JSON file(s).

    Args:
        paths: Path to file, directory, or list of paths
        max_tokens: Maximum total tokens allowed
        min_tokens: Minimum total tokens required
        pattern: Glob pattern for directory search (default: "*.json")
        exclude_pattern: Optional pattern to exclude files
        in_place: If True, modify files in place
        output_dir: Optional directory to save filtered files

    Returns:
        For single file: The filtered annotation data dict
        For multiple files: Number of files processed
    """
    # Normalize input to list of files
    files = _normalize_paths(paths, pattern, exclude_pattern)

    # Process files
    single_file = len(files) == 1
    processed = 0
    last_data = None

    for file_path in tqdm(files, desc="Filtering annotations", disable=single_file):
        try:
            with open(file_path) as f:
                data = json.load(f)

            if "annotations" not in data:
                if single_file:
                    raise ValueError(f"No 'annotations' field found in {file_path}")
                continue

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
            last_data = data

            # Save the file
            if in_place:
                with open(file_path, "w") as f:
                    json.dump(data, f, indent=2)
            elif output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / file_path.name
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

            processed += 1

        except Exception as e:
            if single_file:
                raise
            print(f"Error processing {file_path}: {e}")

    return last_data if single_file else processed


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
        print("  stats <path> - Get annotation statistics")
        print("  reorder <path> - Reorder annotations to standard order")
        print("  remove <path> <model> - Remove a model from file(s)")
        print("  export <directory> <output.csv> - Export to CSV")
        print("\nPath can be a file, directory, or multiple files")
        sys.exit(1)

    command = sys.argv[1]

    if command == "stats":
        if len(sys.argv) < 3:
            print("Usage: python annotation_tools.py stats <path>")
            sys.exit(1)
        stats = get_annotation_stats(sys.argv[2])
        print(f"Files processed: {stats['files_processed']}")
        print(f"Total annotations: {stats['total_annotations']}")
        print("Model counts:")
        for model, count in sorted(stats["model_counts"].items()):
            print(f"  {model}: {count}")

    elif command == "reorder":
        if len(sys.argv) < 3:
            print("Usage: python annotation_tools.py reorder <path>")
            sys.exit(1)
        model_order = [
            "qwen2.5vl:7b",
            "qwen2.5vl:32b",
            "gemma3:4b",
            "gemma3:12b",
            "gemma3:27b",
            "mistral-small3.2:24b",
        ]
        result = reorder_annotations(sys.argv[2], model_order, pattern="shared*.json")
        if isinstance(result, int):
            print(f"Reordered {result} files")
        else:
            print("File reordered successfully")

    elif command == "remove":
        if len(sys.argv) < 4:
            print("Usage: python annotation_tools.py remove <path> <model>")
            sys.exit(1)
        result = remove_model(sys.argv[2], sys.argv[3], pattern="shared*.json")
        if isinstance(result, int):
            print(f"Processed {result} files")
        else:
            print("Model removed from file")

    elif command == "export":
        if len(sys.argv) < 4:
            print("Usage: python annotation_tools.py export <directory> <output.csv>")
            sys.exit(1)
        export_to_csv(sys.argv[2], sys.argv[3], pattern="shared*.json")
        print(f"Exported to {sys.argv[3]}")
