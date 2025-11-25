#!/bin/bash
# Example usage of visualize_gt_annotations.py

echo "======================================"
echo "GT Annotation Visualization Examples"
echo "======================================"
echo ""

# Example 1: Visualize 20 random validation samples
echo "Example 1: Visualize 20 validation samples from split1"
python scripts/visualize_gt_annotations.py \
    --split val \
    --data-split split1 \
    --num-samples 20 \
    --output-dir outputs/gt_visualizations

echo ""

# Example 2: Visualize test set
echo "Example 2: Visualize 30 test samples"
python scripts/visualize_gt_annotations.py \
    --split test \
    --data-split split1 \
    --num-samples 30 \
    --output-dir outputs/gt_visualizations

echo ""

# Example 3: Visualize specific category (e.g., goldenretriever_face, cat_id=48)
echo "Example 3: Visualize only goldenretriever_face (category 48)"
python scripts/visualize_gt_annotations.py \
    --split val \
    --category 48 \
    --num-samples 10 \
    --output-dir outputs/gt_visualizations

echo ""

# Example 4: Visualize all validation categories (split into separate folders)
echo "Example 4: Visualize all val categories separately"
for cat_id in 6 12 22 35 48 66 91 92 95 96; do
    echo "  Processing category $cat_id..."
    python scripts/visualize_gt_annotations.py \
        --split val \
        --category $cat_id \
        --num-samples 5 \
        --output-dir outputs/gt_visualizations
done

echo ""
echo "✅ Done! Check outputs/gt_visualizations/"

