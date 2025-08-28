#!/usr/bin/env python
"""Process NSD dataset with multiple VLM models.

This is the DEFAULT and RECOMMENDED script for processing the NSD dataset.

Processing order: Model -> Images -> Prompts
Benefits:
- Minimizes expensive model loading/unloading operations
- Much faster for multiple models (loads each model only once)
- Better for large-scale processing

For image-by-image processing (all models per image), use process_nsd_by_image.py
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
            return json.load(f)
    return {"completed": {}, "last_checkpoint": None}


def save_progress(progress_file: Path, completed: dict[str, list[str]]) -> None:
    """Save progress checkpoint.

    Args:
        progress_file: Path to progress file
        completed: Dictionary of model -> list of completed image names
    """
    progress = {
        "completed": completed,
        "last_checkpoint": datetime.now(UTC).isoformat(),
        "total_processed": sum(len(images) for images in completed.values()),
    }
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def load_or_create_image_results(output_file: Path) -> dict[str, Any]:
    """Load existing results file or create new structure.

    Args:
        output_file: Path to the JSON results file

    Returns:
        Dictionary with image results structure
    """
    if output_file.exists():
        with open(output_file) as f:
            return json.load(f)
    else:
        return {
            "image": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "annotations": [],
        }


def save_image_results(  # noqa: C901
    image_path: Path, model: str, prompt_results: list[VLMResult], output_dir: Path
) -> Path:
    """Save or append results for a single image-model combination.

    Args:
        image_path: Path to the image file
        model: Model name
        prompt_results: List of VLM results for all prompts
        output_dir: Directory to save results

    Returns:
        Path to the saved JSON file
    """
    # Create output filename based on image name
    image_name = image_path.stem  # e.g., "shared0001_nsd02951"
    output_file = output_dir / f"{image_name}_annotations.json"

    # Load or create results structure
    output_data = load_or_create_image_results(output_file)

    # Update image metadata if not set
    if not output_data["image"]:
        output_data["image"] = {
            "path": str(image_path),
            "name": image_path.name,
            "id": image_name,
        }

    # Create model results
    model_data = {
        "model": model,
        "temperature": prompt_results[0].temperature if prompt_results else 0.3,
        "prompts": {},
        "metrics": {
            "total_tokens": 0,
            "total_time_ms": 0,
            "average_speed": 0,
        },
    }

    # Add prompt results
    for result in prompt_results:
        model_data["prompts"][result.prompt_id] = {
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
            model_data["metrics"]["total_tokens"] += result.token_metrics.total_tokens
        if result.performance_metrics and result.performance_metrics.total_duration_ms:
            model_data["metrics"]["total_time_ms"] += result.performance_metrics.total_duration_ms

    # Calculate average speed
    speeds = []
    for prompt_data in model_data["prompts"].values():
        if (
            prompt_data["performance_metrics"]
            and prompt_data["performance_metrics"]["tokens_per_second"]
        ):
            speeds.append(prompt_data["performance_metrics"]["tokens_per_second"])
    if speeds:
        model_data["metrics"]["average_speed"] = sum(speeds) / len(speeds)

    # Check if this model already exists and update or append
    existing_model_idx = None
    for idx, existing in enumerate(output_data["annotations"]):
        if existing["model"] == model:
            existing_model_idx = idx
            break

    if existing_model_idx is not None:
        output_data["annotations"][existing_model_idx] = model_data
    else:
        output_data["annotations"].append(model_data)

    # Update timestamp
    output_data["timestamp"] = datetime.now(UTC).isoformat()

    # Save to file
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    return output_file


def process_model_batch(
    model: str,
    image_files: list[Path],
    prompts: list[VLMPrompt],
    output_dir: Path,
    temperature: float = 0.3,
    completed_images: set[str] = None,
) -> list[str]:
    """Process all images for a single model.

    This approach minimizes model loading/unloading overhead.

    Args:
        model: Model name to use
        image_files: List of image paths to process
        prompts: List of prompts to run
        output_dir: Directory to save results
        temperature: Generation temperature
        completed_images: Set of already completed image names for this model

    Returns:
        List of successfully processed image names
    """
    if completed_images is None:
        completed_images = set()

    print(f"\n{'=' * 80}")
    print(f"Processing with model: {model}")
    print(f"{'=' * 80}")

    # Filter out already completed images
    images_to_process = [img for img in image_files if img.name not in completed_images]

    if not images_to_process:
        print(f"All images already processed for {model}")
        return list(completed_images)

    print(
        f"Images to process: {len(images_to_process)} (skipping {len(completed_images)} already completed)"
    )

    # Create service once for this model
    service = VLMService(
        model=model,
        temperature=temperature,
        timeout=120,  # Increase timeout for larger models
    )

    processed = list(completed_images)

    # Process each image
    for img_idx, image_path in enumerate(images_to_process, 1):
        print(f"\n[{img_idx}/{len(images_to_process)}] Image: {image_path.name}")
        print("-" * 40)

        prompt_results = []

        # Process each prompt for this image
        for prompt_idx, prompt in enumerate(prompts, 1):
            print(f"  Prompt {prompt_idx}/{len(prompts)}: {prompt.id}...", end=" ")

            try:
                # CRITICAL: Each annotate_image call is stateless
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

                prompt_results.append(result)

            except Exception as e:
                print(f"❌ Failed: {e}")
                # Create error result
                prompt_results.append(
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

        # Save results for this image-model combination
        try:
            output_file = save_image_results(image_path, model, prompt_results, output_dir)
            print(f"  💾 Saved to: {output_file.name}")
            processed.append(image_path.name)
        except Exception as e:
            print(f"  ❌ Failed to save results: {e}")

    return processed


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Process NSD dataset with VLM models (optimized)")
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
        default=[
            "gemma3:4b",
            "gemma3:12b",
            "gemma3:27b",
            "qwen2.5vl:7b",
            "qwen2.5vl:32b",
            "llama3.2-vision:11b",
            "mistral-small3.2:24b",
        ],
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
    progress_file = args.output_dir / "progress_optimized.json"
    progress = load_progress(progress_file) if args.resume else {"completed": {}}

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

    # Create prompts
    prompts = create_comprehensive_prompts()

    print("\nConfiguration:")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Images to process: {len(image_files)}")
    print(f"  Output directory: {args.output_dir}")
    print("\n⚡ Using optimized processing: Model -> Images -> Prompts")
    print("  This minimizes expensive model switching operations")

    # Process each model
    print(f"\n{'=' * 80}")
    print(f"Starting processing at {datetime.now(UTC).isoformat()}")
    print(f"{'=' * 80}")

    completed_all = dict(progress.get("completed", {}))

    for model_idx, model in enumerate(args.models, 1):
        print(f"\n[Model {model_idx}/{len(args.models)}]")

        try:
            # Get already completed images for this model
            completed_for_model = set(completed_all.get(model, []))

            # Process all images with this model
            processed = process_model_batch(
                model=model,
                image_files=image_files,
                prompts=prompts,
                output_dir=args.output_dir,
                temperature=args.temperature,
                completed_images=completed_for_model,
            )

            # Update progress
            completed_all[model] = processed
            save_progress(progress_file, completed_all)

            print(f"\n✅ Completed {len(processed)} images with {model}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Processing interrupted by user")
            save_progress(progress_file, completed_all)
            total_processed = sum(len(images) for images in completed_all.values())
            print(f"Progress saved. Total annotations: {total_processed}")
            break
        except Exception as e:
            print(f"\n❌ Error with model {model}: {e}")
            continue

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"Processing complete at {datetime.now(UTC).isoformat()}")
    print("\nSummary by model:")
    for model, images in completed_all.items():
        print(f"  {model}: {len(images)} images")
    total_annotations = sum(len(images) for images in completed_all.values())
    print(f"\nTotal annotations: {total_annotations}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
