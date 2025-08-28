#!/usr/bin/env python3
"""Generate thumbnails for NSD images."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def generate_thumbnail(image_path: Path, output_path: Path, size: tuple = (150, 150)):
    """Generate a thumbnail for a single image."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=85)
        return True, image_path.name
    except Exception as e:
        return False, f"{image_path.name}: {str(e)}"


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    source_dir = project_root / "images" / "downsampled"
    thumb_dir = project_root / "data" / "thumbnails"

    # Create thumbnail directory
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Get all jpg files
    image_files = list(source_dir.glob("*.jpg"))

    if not image_files:
        print(f"No images found in {source_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for img_path in image_files:
            output_path = thumb_dir / img_path.name
            future = executor.submit(generate_thumbnail, img_path, output_path)
            futures.append(future)

        # Track progress
        succeeded = 0
        failed = []

        with tqdm(total=len(futures), desc="Generating thumbnails") as pbar:
            for future in as_completed(futures):
                success, result = future.result()
                if success:
                    succeeded += 1
                else:
                    failed.append(result)
                pbar.update(1)

    print(f"\nCompleted: {succeeded} thumbnails generated")
    if failed:
        print(f"Failed: {len(failed)} images")
        for error in failed[:5]:  # Show first 5 errors
            print(f"  - {error}")


if __name__ == "__main__":
    main()
