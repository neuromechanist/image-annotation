#!/bin/bash

# Script to update annotations on the production site
# Usage: ./update-annotations.sh

echo "📦 Copying annotations to frontend public directory..."

# Copy annotations
cp -r annotations/nsd/*.json frontend/public/annotations/nsd/ 2>/dev/null && echo "✓ Annotations copied"

# Update image list if thumbnails changed
if [ "$1" == "--update-images" ]; then
    echo "📸 Updating image list..."
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