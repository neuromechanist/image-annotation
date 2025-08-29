#!/usr/bin/env python
"""Smart resume script for NSD processing.

This script examines the progress files and annotation outputs to determine
what needs to be reprocessed, providing clear options for resuming.
"""

import argparse
import json
from pathlib import Path


def analyze_progress(progress_file: Path) -> dict:
    """Analyze a progress file and return summary."""
    if not progress_file.exists():
        return {
            "exists": False,
            "models": {},
            "total_completed": 0,
            "total_partial": 0,
            "total_failed": 0,
        }

    with open(progress_file) as f:
        data = json.load(f)

    result = {
        "exists": True,
        "models": {},
        "total_completed": 0,
        "total_partial": 0,
        "total_failed": 0,
        "last_checkpoint": data.get("last_checkpoint", "Unknown"),
        "current_model": data.get("current_model"),
        "current_image": data.get("current_image"),
    }

    # Handle process_nsd_dataset.py format
    if "completed" in data and isinstance(data["completed"], dict):
        for model, images in data["completed"].items():
            if isinstance(images, dict):
                # New format with status
                completed = [img for img, status in images.items() if status == "completed"]
                partial = [img for img, status in images.items() if status == "partial"]
                failed = [img for img, status in images.items() if status == "failed"]
                result["models"][model] = {
                    "completed": len(completed),
                    "partial": len(partial),
                    "failed": len(failed),
                    "total": len(images),
                }
                result["total_completed"] += len(completed)
                result["total_partial"] += len(partial)
                result["total_failed"] += len(failed)
            else:
                # Old format (list)
                result["models"][model] = {
                    "completed": len(images),
                    "partial": 0,
                    "failed": 0,
                    "total": len(images),
                }
                result["total_completed"] += len(images)

    # Handle process_nsd_by_image.py format
    elif "completed_images" in data:
        images = data["completed_images"]
        if isinstance(images, dict):
            completed = [img for img, status in images.items() if status == "completed"]
            partial = [img for img, status in images.items() if status == "partial"]
            failed = [img for img, status in images.items() if status == "failed"]
            result["total_completed"] = len(completed)
            result["total_partial"] = len(partial)
            result["total_failed"] = len(failed)
        else:
            # Old format (list)
            result["total_completed"] = len(images)

    return result


def analyze_annotations(annotation_dir: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Analyze annotation files to determine what's been processed.

    Returns:
        Tuple of (processed_images, model_to_images_dict)
    """
    processed_images = set()
    model_images = {}

    if not annotation_dir.exists():
        return processed_images, model_images

    for json_file in annotation_dir.glob("*_annotations.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            image_name = json_file.stem.replace("_annotations", "") + ".png"
            processed_images.add(image_name)

            # Track which models have processed this image
            for annotation in data.get("annotations", []):
                model = annotation["model"]
                if model not in model_images:
                    model_images[model] = set()
                model_images[model].add(image_name)
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}")

    return processed_images, model_images


def main():  # noqa: C901
    """Main analysis and recommendation function."""
    parser = argparse.ArgumentParser(description="Analyze and resume NSD processing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotations/nsd"),
        help="Directory containing annotations and progress files",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("images/original"),
        help="Directory containing NSD images",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix inconsistencies in progress files",
    )

    args = parser.parse_args()

    # Check for progress files
    progress_dataset = args.output_dir / "progress_optimized.json"
    progress_image = args.output_dir / "progress.json"

    # Analyze progress files
    dataset_info = analyze_progress(progress_dataset)
    image_info = analyze_progress(progress_image)

    # Analyze actual annotation files
    processed_images, model_images = analyze_annotations(args.output_dir)

    # Count total images available
    total_images = len(list(args.input_dir.glob("*.png")))

    print("=" * 80)
    print("NSD Processing Status Analysis")
    print("=" * 80)

    print(f"\n📁 Total images available: {total_images}")
    print(f"📁 Annotation files found: {len(processed_images)}")

    # Report on process_nsd_dataset.py status
    if dataset_info["exists"]:
        print("\n📊 process_nsd_dataset.py progress:")
        print(f"   Last checkpoint: {dataset_info['last_checkpoint']}")
        if dataset_info["current_model"]:
            print(
                f"   Currently processing: {dataset_info['current_model']} - {dataset_info['current_image']}"
            )
        print(f"   Total completed: {dataset_info['total_completed']}")
        if dataset_info["total_partial"] > 0:
            print(f"   Total partial: {dataset_info['total_partial']}")
        if dataset_info["total_failed"] > 0:
            print(f"   Total failed: {dataset_info['total_failed']}")

        print("\n   Per-model status:")
        for model, stats in dataset_info["models"].items():
            status = f"   - {model}: {stats['completed']}/{stats['total']} completed"
            if stats["partial"] > 0:
                status += f", {stats['partial']} partial"
            if stats["failed"] > 0:
                status += f", {stats['failed']} failed"
            print(status)
    else:
        print("\n📊 process_nsd_dataset.py: No progress file found")

    # Report on process_nsd_by_image.py status
    if image_info["exists"]:
        print("\n📊 process_nsd_by_image.py progress:")
        print(f"   Last checkpoint: {image_info['last_checkpoint']}")
        if image_info["current_image"]:
            print(f"   Currently processing: {image_info['current_image']}")
        print(f"   Images completed: {image_info['total_completed']}")
        if image_info["total_partial"] > 0:
            print(f"   Images partial: {image_info['total_partial']}")
        if image_info["total_failed"] > 0:
            print(f"   Images failed: {image_info['total_failed']}")
    else:
        print("\n📊 process_nsd_by_image.py: No progress file found")

    # Check for inconsistencies
    print("\n🔍 Checking for inconsistencies...")
    issues = []

    # Check if annotation files exist without progress tracking
    for _image in processed_images:
        # Check dataset progress
        if dataset_info["exists"]:
            for model_data in dataset_info["models"].values():
                if isinstance(model_data, dict):
                    continue  # New format, can't easily check

        # Check image progress
        if image_info["exists"] and image_info["total_completed"] > 0:
            pass

    if not issues:
        print("   ✅ No inconsistencies found")
    else:
        for issue in issues:
            print(f"   ⚠️ {issue}")

    # Recommendations
    print("\n💡 Recommendations:")

    if dataset_info["exists"] and any(
        stats["completed"] < total_images for stats in dataset_info["models"].values()
    ):
        print("\n   To resume model-by-model processing (faster for multiple models):")
        print("   python scripts/process_nsd_dataset.py --resume")

        # Show which models need more processing
        incomplete_models = [
            model
            for model, stats in dataset_info["models"].items()
            if stats["completed"] < total_images
        ]
        if incomplete_models:
            print(f"   Models with incomplete processing: {', '.join(incomplete_models)}")

    if image_info["exists"] and image_info["total_completed"] < total_images:
        print("\n   To resume image-by-image processing:")
        print("   python scripts/process_nsd_by_image.py --resume")
        remaining = total_images - image_info["total_completed"]
        print(f"   Images remaining: {remaining}")

    if not dataset_info["exists"] and not image_info["exists"]:
        print("\n   No progress files found. Start fresh with:")
        print("   python scripts/process_nsd_dataset.py  # For model-by-model (faster)")
        print("   python scripts/process_nsd_by_image.py  # For image-by-image")

    # Fix option
    if args.fix and (dataset_info["total_failed"] > 0 or image_info["total_failed"] > 0):
        print("\n🔧 Fixing progress files...")

        # Reset failed items to allow retry
        if dataset_info["exists"] and dataset_info["total_failed"] > 0:
            with open(progress_dataset) as f:
                data = json.load(f)

            for model, images in data["completed"].items():
                if isinstance(images, dict):
                    # Remove failed entries to allow retry
                    data["completed"][model] = {
                        img: status for img, status in images.items() if status != "failed"
                    }

            with open(progress_dataset, "w") as f:
                json.dump(data, f, indent=2)
            print(f"   Fixed {progress_dataset.name}")

        if image_info["exists"] and image_info["total_failed"] > 0:
            with open(progress_image) as f:
                data = json.load(f)

            if isinstance(data["completed_images"], dict):
                # Remove failed entries to allow retry
                data["completed_images"] = {
                    img: status
                    for img, status in data["completed_images"].items()
                    if status != "failed"
                }

            with open(progress_image, "w") as f:
                json.dump(data, f, indent=2)
            print(f"   Fixed {progress_image.name}")

        print("   ✅ Failed items removed, ready to retry")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
