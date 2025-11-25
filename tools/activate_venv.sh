#!/bin/bash
# Convenience script to activate the virtual environment
# Usage: source activate_venv.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/../venv/bin/activate"

echo "✅ Virtual environment activated!"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Available commands:"
echo "  - python -m models.train_cape_episodic --help"
echo "  - python tests/test_checkpoint_system.py"
echo ""
echo "To deactivate: deactivate"

