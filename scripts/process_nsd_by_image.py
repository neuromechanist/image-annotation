#!/usr/bin/env python
"""Process NSD dataset with multiple VLM models - image-by-image approach.

This script processes each image with ALL models before moving to the next image.

Processing order: Image -> Models -> Prompts
Use cases:
- When you want complete annotations for specific images immediately
- When testing or debugging with a few images
- When you may interrupt processing frequently

Note: This approach causes more model switching overhead.
For faster processing of the full dataset, use process_nsd_dataset.py
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_annotation.services.vlm_service import VLMPrompt, VLMResult, VLMService


def create_comprehensive_prompts() -> list[VLMPrompt]:
    """Create the comprehensive set of prompts for NSD annotation."""
    return [
        VLMPrompt(
            id="general_description",
            text=(
                "Describe what you see in this image. Include the setting, main elements, "
                "colors, lighting, and overall composition. Be specific and detailed. "
                "Form the response as a continuous paragraph. Maximum 200 words."
            ),
            expected_format="text",
        ),
        VLMPrompt(
            id="foreground_background",
            text=(
                "Describe the foreground elements (closest to viewer) and background elements "
                "(distant/setting) separately. Explain what appears in each layer and how they "
                "relate to each other. Form the response as a continuous paragraph. Maximum 200 words."
            ),
            expected_format="text",
        ),
        VLMPrompt(
            id="entities_interactions",
            text=(
                "Identify all people, animals, and objects in the image. Describe what they are "
                "doing and how they interact with each other. Focus on actions, relationships, "
                "and connections between elements. Form the response as a continuous paragraph. "
                "Maximum 200 words."
            ),
            expected_format="text",
        ),
        VLMPrompt(
            id="mood_emotions",
            text=(
                "Describe the mood and emotions conveyed by this image. What feelings does it evoke? "
                "Consider whether the overall tone is positive, negative, or neutral. Explain what "
                "visual elements contribute to this emotional atmosphere. Form the response as a "
                "continuous paragraph. Maximum 200 words."
            ),
            expected_format="text",
        ),
        VLMPrompt(
            id="structured_inventory",
            text=(
                "Analyze this image and create a JSON object documenting all visible items. "
                "Structure the output with these exact three levels:\n"
                "Level 1 (Categories): Use only these four keys: 'human', 'animal', 'man-made', 'natural'\n"
                "Level 2 (Item names): Specific names of detected items (e.g., 'person', 'dog', 'car', 'tree')\n"
                "Level 3 (Attributes): Use ONLY these keys for each item:\n"
                "  • count: number of instances (integer)\n"
                "  • location: position in image (use terms like: left/center/right, top/middle/bottom, foreground/background)\n"
                "  • color: main color(s) if applicable (array of strings)\n"
                "  • size: relative size (small/medium/large)\n"
                "  • description: any other relevant details that don't fit above categories (string)\n"
                "Output valid JSON only. Include only categories that contain detected items. "
                "If an attribute doesn't apply to an item (e.g., color for sky), omit that key "
                "rather than using null. The 'description' field should capture important "
                "characteristics like actions, states, or specific features not covered by other keys."
            ),
            expected_format="json",
        ),
    ]


def load_progress(progress_file: Path) -> dict[str, Any]:
    """Load existing progress from checkpoint file."""
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
            # Handle old format (list) and new format (dict with per-image status)
            if "completed_images" in data:
                if isinstance(data["completed_images"], list):
                    # Old format: convert list to dict
                    data["completed_images"] = {
                        img: "completed" for img in data["completed_images"]
                    }
            return data
    return {"completed_images": {}, "last_checkpoint": None, "current_image": None}


def save_progress(
    progress_file: Path, completed_images: dict[str, str], current_image: str = None
) -> None:
    """Save progress checkpoint with per-image status tracking.

    Args:
        progress_file: Path to progress file
        completed_images: Dict of image_name -> status
        current_image: Currently processing image
    """
    # Count completed images
    total_completed = len(
        [img for img, status in completed_images.items() if status == "completed"]
    )

    progress = {
        "completed_images": completed_images,
        "last_checkpoint": datetime.now(UTC).isoformat(),
        "total_processed": total_completed,
        "current_image": current_image,
    }
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def save_image_results(image_path: Path, results: list[VLMResult], output_dir: Path) -> Path:
    """Save results for a single image to a JSON file.

    Args:
        image_path: Path to the image file
        results: List of VLM results for this image
        output_dir: Directory to save results

    Returns:
        Path to the saved JSON file
    """
    # Create output filename based on image name
    image_name = image_path.stem  # e.g., "shared0001_nsd02951"
    output_file = output_dir / f"{image_name}_annotations.json"

    # Structure the results
    output_data = {
        "image": {
            "path": str(image_path),
            "name": image_path.name,
            "id": image_name,
        },
        "timestamp": datetime.now(UTC).isoformat(),
        "annotations": [],
    }

    # Group results by model
    model_results = {}
    for result in results:
        if result.model not in model_results:
            model_results[result.model] = {
                "model": result.model,
                "temperature": result.temperature,
                "prompts": {},
                "metrics": {
                    "total_tokens": 0,
                    "total_time_ms": 0,
                    "average_speed": 0,
                },
            }

        # Add prompt result
        model_results[result.model]["prompts"][result.prompt_id] = {
            "prompt_text": result.prompt_text,
            "response": result.response,
            "response_format": result.response_format,
            "response_data": result.response_data,
            "error": result.error,
            "token_metrics": result.token_metrics.model_dump() if result.token_metrics else None,
            "performance_metrics": result.performance_metrics.model_dump()
            if result.performance_metrics
            else None,
        }

        # Update aggregate metrics
        if result.token_metrics and result.token_metrics.total_tokens:
            model_results[result.model]["metrics"]["total_tokens"] += (
                result.token_metrics.total_tokens
            )
        if result.performance_metrics and result.performance_metrics.total_duration_ms:
            model_results[result.model]["metrics"]["total_time_ms"] += (
                result.performance_metrics.total_duration_ms
            )

    # Calculate average speeds
    for model_data in model_results.values():
        speeds = []
        for prompt_data in model_data["prompts"].values():
            if (
                prompt_data["performance_metrics"]
                and prompt_data["performance_metrics"]["tokens_per_second"]
            ):
                speeds.append(prompt_data["performance_metrics"]["tokens_per_second"])
        if speeds:
            model_data["metrics"]["average_speed"] = sum(speeds) / len(speeds)

    output_data["annotations"] = list(model_results.values())

    # Save to file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    return output_file


def process_single_image(  # noqa: C901
    image_path: Path,
    models: list[str],
    prompts: list[VLMPrompt],
    output_dir: Path,
    temperature: float = 0.3,
    check_existing: bool = True,
) -> tuple[Path, bool]:
    """Process a single image with all models and prompts.

    IMPORTANT: Ensures stateless processing - each prompt gets a fresh context.

    Args:
        image_path: Path to the image file
        models: List of model names to use
        prompts: List of prompts to run
        output_dir: Directory to save results
        temperature: Generation temperature
        check_existing: Whether to check and load existing partial results

    Returns:
        Tuple of (Path to saved results file, success status)
    """
    print(f"\n{'=' * 80}")
    print(f"Processing: {image_path.name}")
    print(f"{'=' * 80}")

    all_results = []

    # Check for existing partial results
    image_name = image_path.stem
    output_file = output_dir / f"{image_name}_annotations.json"
    existing_models = set()

    if check_existing and output_file.exists():
        try:
            with open(output_file) as f:
                existing_data = json.load(f)
                existing_models = {ann["model"] for ann in existing_data.get("annotations", [])}
                if existing_models:
                    print(f"Found existing annotations for models: {', '.join(existing_models)}")
                    # Load existing results
                    for ann in existing_data["annotations"]:
                        for prompt_id, prompt_data in ann["prompts"].items():
                            all_results.append(
                                VLMResult(
                                    image_path=str(image_path),
                                    model=ann["model"],
                                    prompt_id=prompt_id,
                                    prompt_text=prompt_data["prompt_text"],
                                    response=prompt_data.get("response", ""),
                                    response_format=prompt_data.get("response_format"),
                                    temperature=ann["temperature"],
                                    error=prompt_data.get("error"),
                                )
                            )
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    # Process each model
    success = True
    for model_idx, model in enumerate(models, 1):
        # Skip models that have already been processed
        if model in existing_models:
            print(f"\nModel {model_idx}/{len(models)}: {model} - Already processed, skipping")
            continue

        print(f"\nModel {model_idx}/{len(models)}: {model}")
        print("-" * 40)

        # Create a NEW service instance for each model to ensure clean state
        service = VLMService(
            model=model,
            temperature=temperature,
            timeout=120,  # Increase timeout for larger models
        )

        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts, 1):
            print(f"  Prompt {prompt_idx}/{len(prompts)}: {prompt.id}...", end=" ")

            try:
                # CRITICAL: Each annotate_image call is stateless
                # The service creates a fresh LangChain conversation for each call
                result = service.annotate_image(image_path, prompt, model)

                if result.error:
                    print(f"❌ Error: {result.error}")
                else:
                    if result.token_metrics and result.performance_metrics:
                        print(
                            f"✅ {result.token_metrics.total_tokens} tokens, "
                            f"{result.performance_metrics.tokens_per_second:.1f} tok/s"
                        )
                    else:
                        print("✅ Complete")

                all_results.append(result)

            except Exception as e:
                print(f"❌ Failed: {e}")
                success = False
                # Create error result
                all_results.append(
                    VLMResult(
                        image_path=str(image_path),
                        model=model,
                        prompt_id=prompt.id,
                        prompt_text=prompt.text,
                        response="",
                        response_format=prompt.expected_format,
                        temperature=temperature,
                        error=str(e),
                    )
                )

    # Save results for this image
    try:
        output_file = save_image_results(image_path, all_results, output_dir)
        print(f"\n✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")
        success = False

    return output_file, success


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Process NSD dataset with VLM models")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("images/original"),
        help="Directory containing NSD images (relative to project root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotations/nsd"),
        help="Directory to save annotations (relative to project root)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma3:4b", "gemma3:12b", "llama3.2-vision:11b", "mistral-small3.2:24b"],
        help="Models to use for annotation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generation temperature (lower = more consistent/reproducible)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this image index",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Progress tracking
    progress_file = args.output_dir / "progress.json"
    progress = load_progress(progress_file) if args.resume else {"completed_images": []}

    # Get list of images
    image_files = sorted(args.input_dir.glob("*.png"))
    if not image_files:
        print(f"No PNG images found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images in {args.input_dir}")

    # Apply start index and max images
    if args.start_index > 0:
        image_files = image_files[args.start_index :]
    if args.max_images:
        image_files = image_files[: args.max_images]

    # Filter out already completed images if resuming
    if args.resume and progress["completed_images"]:
        completed_set = {
            img for img, status in progress["completed_images"].items() if status == "completed"
        }
        image_files = [img for img in image_files if img.name not in completed_set]
        print(f"Resuming: {len(completed_set)} already processed, {len(image_files)} remaining")

    # Create prompts
    prompts = create_comprehensive_prompts()

    print("\nConfiguration:")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Images to process: {len(image_files)}")
    print(f"  Output directory: {args.output_dir}")

    # Process images
    print(f"\n{'=' * 80}")
    print(f"Starting processing at {datetime.now(UTC).isoformat()}")
    print(f"{'=' * 80}")

    completed = dict(progress.get("completed_images", {}))

    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{len(image_files)}] Processing {image_path.name}")

            # Update progress to mark current image as in-progress
            save_progress(progress_file, completed, image_path.name)

            # Process the image with all models and prompts
            # IMPORTANT: Each image gets a completely fresh start
            output_file, success = process_single_image(
                image_path=image_path,
                models=args.models,
                prompts=prompts,
                output_dir=args.output_dir,
                temperature=args.temperature,
                check_existing=args.resume,  # Check for partial results if resuming
            )

            # Update progress based on success
            if success:
                completed[image_path.name] = "completed"
            else:
                completed[image_path.name] = "partial"
            save_progress(progress_file, completed)

        except KeyboardInterrupt:
            print("\n\n⚠️ Processing interrupted by user")
            save_progress(progress_file, completed)
            completed_count = len(
                [img for img, status in completed.items() if status == "completed"]
            )
            partial_count = len([img for img, status in completed.items() if status == "partial"])
            print(f"Progress saved. Completed: {completed_count}, Partial: {partial_count}")
            break
        except Exception as e:
            print(f"\n❌ Error processing {image_path.name}: {e}")
            completed[image_path.name] = "failed"
            save_progress(progress_file, completed)
            continue

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"Processing complete at {datetime.now(UTC).isoformat()}")
    completed_count = len([img for img, status in completed.items() if status == "completed"])
    partial_count = len([img for img, status in completed.items() if status == "partial"])
    failed_count = len([img for img, status in completed.items() if status == "failed"])
    print(
        f"Total images - Completed: {completed_count}, Partial: {partial_count}, Failed: {failed_count}"
    )
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
