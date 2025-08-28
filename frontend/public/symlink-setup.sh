#!/bin/bash
# Create symbolic links to serve static files in Next.js public directory

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Create symlinks in public directory
ln -sf "$PROJECT_ROOT/data" "$SCRIPT_DIR/data"
ln -sf "$PROJECT_ROOT/images" "$SCRIPT_DIR/images"
ln -sf "$PROJECT_ROOT/annotations" "$SCRIPT_DIR/annotations"

echo "Symlinks created:"
ls -la "$SCRIPT_DIR" | grep '^l'