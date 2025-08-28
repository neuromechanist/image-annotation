#!/bin/bash

# Script to update annotations on the production site
# Usage: ./update-annotations.sh [--all]
#        --all: Also update images (thumbnails and downsampled)

echo "🚀 Updating The Annotation Garden Project..."

# Copy annotations
echo "📦 Copying annotations..."
mkdir -p frontend/public/annotations/nsd
cp -r annotations/nsd/*.json frontend/public/annotations/nsd/ 2>/dev/null && echo "✓ Annotations copied"

# Update all assets if requested
if [ "$1" == "--all" ]; then
    echo "📸 Updating images..."
    
    # Copy thumbnails
    mkdir -p frontend/public/thumbnails
    cp -r data/thumbnails/* frontend/public/thumbnails/ 2>/dev/null && echo "✓ Thumbnails copied"
    
    # Copy downsampled images
    mkdir -p frontend/public/downsampled
    cp -r images/downsampled/* frontend/public/downsampled/ 2>/dev/null && echo "✓ Downsampled images copied"
    
    # Update image list
    echo "📝 Generating image list..."
    echo '{"images": [' > frontend/public/image-list.json
    ls frontend/public/thumbnails/ | sed 's/.jpg$//' | awk '{print "  \""$1"\","}' | sed '$ s/,$//' >> frontend/public/image-list.json
    echo ']}' >> frontend/public/image-list.json
    echo "✓ Image list updated"
fi

echo "📝 Committing changes..."
git add frontend/public/annotations/
git add frontend/public/image-list.json 2>/dev/null

git commit -m "Update annotations

Updated NSD annotations with latest model responses"

echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Done! The changes will be deployed automatically in ~2 minutes."
echo "📍 Check deployment status: https://github.com/neuromechanist/hed-image-annotation/actions"