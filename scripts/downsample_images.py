#!/usr/bin/env python3
"""
Downsample images for web display.
Converts original NSD images to optimized versions for the dashboard.
"""

import argparse
from pathlib import Path

from PIL import Image


def downsample_image(
    input_path: Path, output_path: Path, max_dimension: int = 800, quality: int = 85
) -> tuple[int, int]:
    """
    Downsample a single image to optimize for web display.

    Args:
        input_path: Path to original image
        output_path: Path to save downsampled image
        max_dimension: Maximum width or height
        quality: JPEG quality (1-100)

    Returns:
        Tuple of (original_size_kb, new_size_kb)
    """
    # Open and get original size
    img = Image.open(input_path)
    original_size = input_path.stat().st_size // 1024  # KB

    # Calculate new dimensions maintaining aspect ratio
    width, height = img.size
    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        # Resize with high-quality resampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to RGB if necessary (for PNG with transparency)
    if img.mode in ("RGBA", "LA", "P"):
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            rgb_img.paste(img, mask=img.split()[3])
        else:
            rgb_img.paste(img)
        img = rgb_img

    # Save as JPEG
    output_path = output_path.with_suffix(".jpg")
    img.save(output_path, "JPEG", quality=quality, optimize=True)

    new_size = output_path.stat().st_size // 1024  # KB
    return original_size, new_size


def main():
    parser = argparse.ArgumentParser(description="Downsample NSD images for web display")
    parser.add_argument(
        "--input-dir", default="images/original", help="Directory with original images"
    )
    parser.add_argument(
        "--output-dir", default="images/downsampled", help="Directory for downsampled images"
    )
    parser.add_argument(
        "--max-dim", type=int, default=800, help="Maximum dimension (width or height) in pixels"
    )
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all images
    image_extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix in image_extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Max dimension: {args.max_dim}px, Quality: {args.quality}%")
    print("-" * 50)

    total_original = 0
    total_new = 0

    for i, img_path in enumerate(image_files, 1):
        output_path = output_dir / img_path.name
        try:
            orig_kb, new_kb = downsample_image(img_path, output_path, args.max_dim, args.quality)

            total_original += orig_kb
            total_new += new_kb

            reduction = (1 - new_kb / orig_kb) * 100
            print(
                f"[{i}/{len(image_files)}] {img_path.name}: "
                f"{orig_kb}KB → {new_kb}KB ({reduction:.1f}% reduction)"
            )

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print("-" * 50)
    print(f"Total: {total_original / 1024:.1f}MB → {total_new / 1024:.1f}MB")
    print(f"Overall reduction: {(1 - total_new / total_original) * 100:.1f}%")

    if total_new / 1024 > 80:
        print(f"⚠️  Warning: Total size {total_new / 1024:.1f}MB exceeds 80MB target")
        print("Consider reducing quality or max dimension")


if __name__ == "__main__":
    main()
