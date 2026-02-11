#!/bin/bash
# Quick Start Guide for Swap Negatives Generation

echo "=========================================="
echo "Swap Negatives Generation - Quick Start"
echo "=========================================="
echo ""

# Change to the correct directory
cd /mnt/lustre/work/oh/owl336/LabCLIP_v2/CLIP-not-BoW-unimodally

# Step 1: Run all tests
echo "STEP 1: Running tests..."
echo "----------------------------------------"
python scripts/test_swap_negatives.py

if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Please fix errors before proceeding."
    exit 1
fi

echo ""
echo "✅ All tests passed!"
echo ""

# Step 2: Dry run speed test on 10 files
echo "STEP 2: Speed test (dry run, no saving)..."
echo "----------------------------------------"
python scripts/add_swap_negatives.py \
    --input_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components \
    --output_dir /tmp/swap_negatives_test \
    --test_first 10 \
    --num_processes 8 \
    --dry_run

if [ $? -ne 0 ]; then
    echo "❌ Speed test failed!"
    exit 1
fi

echo ""
echo "✅ Speed test complete!"
echo ""

# Step 3: Ask user if they want to proceed
echo "=========================================="
echo "Ready to process all files?"
echo "=========================================="
echo ""
echo "This will:"
echo "  - Process all JSON files in laion400m_negatives_components"
echo "  - Add 'swap_negatives' field to each sample"
echo "  - Save to laion400m_negatives_components_with_swaps"
echo "  - Use 8 parallel processes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 4: Process all files
echo ""
echo "STEP 3: Processing all files..."
echo "----------------------------------------"
python scripts/add_swap_negatives.py \
    --input_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components \
    --output_dir /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps \
    --num_processes 8

if [ $? -ne 0 ]; then
    echo "❌ Processing failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 All done!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  /mnt/lustre/work/oh/owl336/LabCLIP_v2/laion400m_negatives_components_with_swaps"
echo ""
