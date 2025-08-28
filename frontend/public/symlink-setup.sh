#!/bin/bash
# Copy static files to Next.js public directory for GitHub Pages compatibility

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Remove old symlinks if they exist
[ -L "$SCRIPT_DIR/data" ] && rm "$SCRIPT_DIR/data"
[ -L "$SCRIPT_DIR/images" ] && rm "$SCRIPT_DIR/images"
[ -L "$SCRIPT_DIR/annotations" ] && rm "$SCRIPT_DIR/annotations"

# Copy directories for GitHub Pages compatibility
echo "Copying static assets..."
cp -r "$PROJECT_ROOT/data/thumbnails" "$SCRIPT_DIR/thumbnails"
cp -r "$PROJECT_ROOT/images/downsampled" "$SCRIPT_DIR/downsampled"
mkdir -p "$SCRIPT_DIR/annotations/nsd"
cp "$PROJECT_ROOT/annotations/nsd/"*.json "$SCRIPT_DIR/annotations/nsd/"

echo "Static files copied for GitHub Pages deployment"