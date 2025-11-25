#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || { echo "Error: Could not change to script directory."; exit 1; }

echo "Running Simple CAPE Visualization..."
echo "Current directory: $(pwd)"
echo ""

# Check if venv exists, if not use system python
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No venv found, using system python..."
fi

# Run simple visualization (ground truth only - no model loading needed!)
python -m models.visualize_results_simple \
    --mode gt \
    --dataset_root . \
    --num_samples 3 \
    --output_dir visualizations/ground_truth

echo ""
echo "✓ Visualization complete!"
echo "Results saved to: visualizations/ground_truth/"
echo ""
echo "To view results:"
echo "  open visualizations/ground_truth/"

