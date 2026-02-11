#!/bin/bash
# Script to download external CLIP checkpoints for evaluation
# Usage: bash download_external_checkpoints.sh

set -e

# Create checkpoints directory
CHECKPOINT_DIR="external_checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "========================================="
echo "Downloading External CLIP Checkpoints"
echo "========================================="
echo

# ============================================================================
# CE-CLIP (Contrastive Entropy CLIP)
# ============================================================================
echo "1. CE-CLIP (le723z/CE_CLIP)"
echo "   Paper: https://arxiv.org/abs/2210.00141"
echo "   Note: Can be loaded directly from HuggingFace hub"
echo "   No download needed - use checkpoint='le723z/CE_CLIP' checkpoint_type='huggingface'"
echo

# ============================================================================
# FSC-CLIP (Few-Shot CLIP)
# ============================================================================
echo "2. FSC-CLIP (ytaek-oh/fsc-clip)"
echo "   Paper: https://arxiv.org/abs/2308.02151"
echo "   Downloading 3 variants..."
echo

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli not found!"
    echo "   Install with: pip install huggingface_hub"
    echo "   Or download manually from: https://huggingface.co/ytaek-oh/fsc-clip"
    exit 1
fi

# Download FSC-CLIP variants
echo "   Downloading laioncoco_fsc-clip-ViT-B-32.pt..."
huggingface-cli download ytaek-oh/fsc-clip laioncoco_fsc-clip-ViT-B-32.pt \
    --local-dir "$CHECKPOINT_DIR/fsc-clip" \
    --local-dir-use-symlinks False

echo "   Downloading coco_fsc-clip-ViT-B-32.pt..."
huggingface-cli download ytaek-oh/fsc-clip coco_fsc-clip-ViT-B-32.pt \
    --local-dir "$CHECKPOINT_DIR/fsc-clip" \
    --local-dir-use-symlinks False

echo "   Downloading cc3m_fsc-clip-ViT-B-32.pt..."
huggingface-cli download ytaek-oh/fsc-clip cc3m_fsc-clip-ViT-B-32.pt \
    --local-dir "$CHECKPOINT_DIR/fsc-clip" \
    --local-dir-use-symlinks False

echo
echo "========================================="
echo "✓ Download Complete!"
echo "========================================="
echo
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo
echo "File structure:"
echo "  $CHECKPOINT_DIR/"
echo "  └── fsc-clip/"
echo "      ├── laioncoco_fsc-clip-ViT-B-32.pt"
echo "      ├── coco_fsc-clip-ViT-B-32.pt"
echo "      └── cc3m_fsc-clip-ViT-B-32.pt"
echo
echo "Usage examples:"
echo
echo "1. Evaluate CE-CLIP (direct from HuggingFace):"
echo "   python evaluate_checkpoint.py --config-name eval_external_checkpoints \\"
echo "       checkpoint=le723z/CE_CLIP checkpoint_type=huggingface"
echo
echo "2. Evaluate FSC-CLIP (LaiOnCoco):"
echo "   python evaluate_checkpoint.py --config-name eval_external_checkpoints \\"
echo "       checkpoint=$CHECKPOINT_DIR/fsc-clip/laioncoco_fsc-clip-ViT-B-32.pt \\"
echo "       checkpoint_type=external base_model=ViT-B/32"
echo
echo "3. Evaluate FSC-CLIP (COCO):"
echo "   python evaluate_checkpoint.py --config-name eval_external_checkpoints \\"
echo "       checkpoint=$CHECKPOINT_DIR/fsc-clip/coco_fsc-clip-ViT-B-32.pt \\"
echo "       checkpoint_type=external base_model=ViT-B/32"
echo
echo "4. Evaluate FSC-CLIP (CC3M):"
echo "   python evaluate_checkpoint.py --config-name eval_external_checkpoints \\"
echo "       checkpoint=$CHECKPOINT_DIR/fsc-clip/cc3m_fsc-clip-ViT-B-32.pt \\"
echo "       checkpoint_type=external base_model=ViT-B/32"
echo
