# Image Storage Directory

## Structure
```
images/
├── original/       # Original NSD images (1000 images from shared1000)
├── downsampled/    # Optimized versions for web display
└── README.md       # This file
```

## Usage

### Original Images
Place the NSD images from `/Volumes/S1/Datasets/NSD_stimuli/shared1000/` here.
These files are not tracked by git (.gitignore configured for *.png, *.jpg, *.jpeg).

### Downsampled Images
Run the downsampling script to generate web-optimized versions:
```bash
python scripts/downsample_images.py
```

## Image Requirements
- Original: Full resolution NSD images
- Downsampled: Max dimension 800px, JPEG quality 85%
- Target total size for downsampled: <80MB for all 1000 images