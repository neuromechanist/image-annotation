#!/bin/bash
# Convenience script to update annotations with smart resume capability

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔄 NSD Annotation Update Script"
echo "================================"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Conda not found. Please install Miniconda first.${NC}"
    exit 1
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch-312

# Check current status
echo -e "\n${YELLOW}📊 Analyzing current progress...${NC}"
python scripts/resume_processing.py

# Ask user what to do
echo -e "\n${GREEN}Select processing mode:${NC}"
echo "1) Resume model-by-model processing (recommended for multiple models)"
echo "2) Resume image-by-image processing (all models per image)"
echo "3) Analyze only (exit)"
echo "4) Fix failed items and retry"
echo -n "Choice [1-4]: "
read choice

case $choice in
    1)
        echo -e "\n${GREEN}Starting model-by-model processing...${NC}"
        python scripts/process_nsd_dataset.py --resume
        ;;
    2)
        echo -e "\n${GREEN}Starting image-by-image processing...${NC}"
        python scripts/process_nsd_by_image.py --resume
        ;;
    3)
        echo -e "\n${YELLOW}Analysis complete. Exiting.${NC}"
        exit 0
        ;;
    4)
        echo -e "\n${YELLOW}Fixing failed items...${NC}"
        python scripts/resume_processing.py --fix
        echo -e "${GREEN}Failed items cleared. Run the script again to retry processing.${NC}"
        ;;
    *)
        echo -e "\n${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

# Show final status
echo -e "\n${YELLOW}📊 Final status:${NC}"
python scripts/resume_processing.py

echo -e "\n${GREEN}✅ Processing complete!${NC}"