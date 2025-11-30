#!/bin/bash
#
# Test K-Fold Setup
#
# This script verifies that k-fold cross-validation is properly configured
# by running a minimal test (1 epoch per split) without full training.
#
# Usage:
#   ./scripts/test_kfold_setup.sh
#

set -e

echo "=============================================================================="
echo "  K-Fold Cross-Validation Setup Test"
echo "=============================================================================="
echo ""
echo "This will run a minimal k-fold test (1 epoch per split) to verify setup."
echo "Estimated time: ~10 minutes"
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Running test..."
echo ""

# Run k-fold with minimal settings
./scripts/run_kfold_cross_validation.sh \
    --epochs 1 \
    --episodes 10 \
    --batch_size 1 \
    --output_dir outputs/kfold_setup_test

echo ""
echo "=============================================================================="
echo "  Setup Test Complete!"
echo "=============================================================================="
echo ""
echo "Check results:"
echo "  cat outputs/kfold_setup_test/kfold_report.txt"
echo ""
echo "If you see aggregated results, your k-fold setup is working correctly!"
echo ""
echo "To run full k-fold cross-validation:"
echo "  ./scripts/run_kfold_cross_validation.sh --epochs 300"
echo ""

