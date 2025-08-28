#!/bin/bash

# Setup symbolic links for local development
# This allows the frontend to read directly from source directories
# without copying files to public/

echo "🔗 Setting up symbolic links for local development..."

cd frontend/public

# Remove existing directories if they exist
rm -rf thumbnails downsampled annotations

# Create symbolic links to source directories
ln -s ../../data/thumbnails thumbnails
ln -s ../../images/downsampled downsampled
ln -s ../../annotations annotations

echo "✅ Symbolic links created!"
echo "📁 Local dev now reads directly from:"
echo "   - data/thumbnails/"
echo "   - images/downsampled/" 
echo "   - annotations/"
echo ""
echo "🚀 You can now run 'npm run dev' without copying files!"