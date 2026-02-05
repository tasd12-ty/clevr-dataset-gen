#!/bin/bash
# Automatic fix script for Blender import issues on Linux
# Usage: ./fix_blender_import_linux.sh [blender_path]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get Blender executable
BLENDER="${1:-blender}"

echo -e "${GREEN}=== Blender Import Fix Script ===${NC}"
echo ""

# Check if Blender exists
if ! command -v "$BLENDER" &> /dev/null; then
    echo -e "${RED}Error: Blender not found at: $BLENDER${NC}"
    echo "Usage: $0 [/path/to/blender]"
    exit 1
fi

# Get Blender version and Python version
echo "Detecting Blender configuration..."
BLENDER_INFO=$("$BLENDER" --version 2>&1 | head -n 1)
echo "  $BLENDER_INFO"

# Get Python version from Blender
PYTHON_VERSION=$("$BLENDER" --background --python-expr "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1 | tail -n 1)
echo "  Python version: $PYTHON_VERSION"

# Get Blender version
BLENDER_VERSION=$("$BLENDER" --background --python-expr "import bpy; print('.'.join(map(str, bpy.app.version[:2])))" 2>&1 | tail -n 1)
echo "  Blender config version: $BLENDER_VERSION"

# Get absolute path to image_generation directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_GEN_DIR="$PROJECT_DIR/image_generation"

if [ ! -d "$IMAGE_GEN_DIR" ]; then
    echo -e "${RED}Error: image_generation directory not found at: $IMAGE_GEN_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Found image_generation at: $IMAGE_GEN_DIR${NC}"

# Determine site-packages directory
SITE_PACKAGES_DIRS=(
    "$HOME/.config/blender/$BLENDER_VERSION/python/lib/python$PYTHON_VERSION/site-packages"
    "$HOME/snap/blender/common/.config/blender/$BLENDER_VERSION/python/lib/python$PYTHON_VERSION/site-packages"
)

SITE_PACKAGES=""
for dir in "${SITE_PACKAGES_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        SITE_PACKAGES="$dir"
        echo -e "${GREEN}Found site-packages at: $SITE_PACKAGES${NC}"
        break
    fi
done

if [ -z "$SITE_PACKAGES" ]; then
    # Create the directory
    SITE_PACKAGES="${SITE_PACKAGES_DIRS[0]}"
    echo -e "${YELLOW}Creating site-packages directory: $SITE_PACKAGES${NC}"
    mkdir -p "$SITE_PACKAGES"
fi

# Create or update clevr.pth
PTH_FILE="$SITE_PACKAGES/clevr.pth"

if [ -f "$PTH_FILE" ]; then
    echo -e "${YELLOW}Updating existing clevr.pth${NC}"
    # Remove old entry if exists
    grep -v "image_generation" "$PTH_FILE" > "$PTH_FILE.tmp" || true
    echo "$IMAGE_GEN_DIR" >> "$PTH_FILE.tmp"
    mv "$PTH_FILE.tmp" "$PTH_FILE"
else
    echo -e "${GREEN}Creating new clevr.pth${NC}"
    echo "$IMAGE_GEN_DIR" > "$PTH_FILE"
fi

echo "  Created: $PTH_FILE"
echo "  Content: $(cat "$PTH_FILE")"

# Verify the fix
echo ""
echo "Verifying import..."
VERIFY_RESULT=$("$BLENDER" --background --python-expr "
import sys
sys.path.insert(0, '$IMAGE_GEN_DIR')
try:
    import utils
    print('SUCCESS: utils imported from', utils.__file__)
except ImportError as e:
    print('FAILED:', e)
" 2>&1 | grep -E "(SUCCESS|FAILED)")

if echo "$VERIFY_RESULT" | grep -q "SUCCESS"; then
    echo -e "${GREEN}✓ Import verification successful!${NC}"
    echo "  $VERIFY_RESULT"
else
    echo -e "${RED}✗ Import verification failed!${NC}"
    echo "  $VERIFY_RESULT"
    echo ""
    echo "Try manually adding the path in your script:"
    echo "  import sys"
    echo "  sys.path.insert(0, '$IMAGE_GEN_DIR')"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Fix Complete ===${NC}"
echo ""
echo "You can now run:"
echo "  cd $IMAGE_GEN_DIR"
echo "  $BLENDER --background --python render_images.py -- --num_images 1"
