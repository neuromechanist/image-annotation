#!/usr/bin/env python
"""Test script to verify stateless processing - no memory/prompt leakage between calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_annotation.services.vlm_service import VLMPrompt, VLMService


def test_stateless_processing(image_path=None):
    """Test that each prompt gets a fresh context with no memory of previous prompts.

    Args:
        image_path: Optional path to test image. Defaults to first downsampled image.
    """

    print("Testing Stateless Processing")
    print("=" * 60)

    # Use a test image - default to first downsampled image
    if image_path is None:
        image_path = Path("images/downsampled/shared0001_nsd02951.jpg")
    else:
        image_path = Path(image_path)

    # Create prompts that would reveal context leakage
    prompts = [
        VLMPrompt(
            id="test1",
            text="Describe this image in exactly 3 words. Remember the word 'elephant' for later.",
            expected_format="text",
        ),
        VLMPrompt(
            id="test2",
            text="What word did I ask you to remember in my previous message? If you don't know, just say 'No previous context'",
            expected_format="text",
        ),
        VLMPrompt(
            id="test3",
            text="List exactly 3 colors you see in this image as a JSON array",
            expected_format="json",
        ),
        VLMPrompt(
            id="test4",
            text="How many colors did I ask you to list in the previous prompt? If you don't know, say 'No previous context'",
            expected_format="text",
        ),
    ]

    # Create service
    service = VLMService(model="gemma3:4b", temperature=0.3)  # Lower temp for consistency

    print(f"Image: {image_path.name}")
    print("Model: gemma3:4b")
    print("-" * 60)

    # Process each prompt
    for prompt in prompts:
        print(f"\nPrompt {prompt.id}: {prompt.text}")
        print("-" * 40)

        result = service.annotate_image(image_path, prompt)

        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Response: {result.response[:200]}")

            # Check for context leakage
            if prompt.id in ["test2", "test4"]:
                response_lower = result.response.lower()
                if any(
                    word in response_lower
                    for word in ["elephant", "remember", "previous", "3 colors", "three colors"]
                ):
                    print("⚠️ WARNING: Possible context leakage detected!")
                else:
                    print("✅ No context leakage - response shows no memory of previous prompts")

    print("\n" + "=" * 60)
    print("Stateless Test Complete")
    print("Each prompt should have received a fresh context.")
    print("Prompts 2 and 4 should indicate 'No previous context' or similar.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test stateless VLM processing")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image (defaults to images/downsampled/shared0001_nsd02951.jpg)",
    )

    args = parser.parse_args()
    test_stateless_processing(args.image)
