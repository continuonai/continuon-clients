#!/bin/bash
# Hailo NPU Compilation Script for Seed Model
# Run this on a machine with Hailo SDK installed

set -e

EXPORT_DIR="${1:-/opt/continuonos/brain/model/exports/hailo}"
OUTPUT_DIR="${2:-/opt/continuonos/brain/model/base_model}"

echo "============================================================"
echo "HAILO NPU COMPILATION"
echo "============================================================"

# Check for Hailo tools
if ! command -v hailo &> /dev/null; then
    echo "❌ Hailo SDK not found. Install from: https://developer.hailo.ai"
    exit 1
fi

cd "$EXPORT_DIR"

echo ""
echo "1️⃣ Creating ONNX model..."
python3 create_onnx.py

echo ""
echo "2️⃣ Optimizing for Hailo-8..."
hailo optimize seed_model.onnx --hw-arch hailo8 --output-har seed_model.har

echo ""
echo "3️⃣ Compiling to HEF..."
hailo compile seed_model.har -o seed_model.hef

echo ""
echo "4️⃣ Installing to runtime..."
mkdir -p "$OUTPUT_DIR"
cp seed_model.hef "$OUTPUT_DIR/model.hef"

echo ""
echo "✅ Compilation complete!"
echo "   HEF: $OUTPUT_DIR/model.hef"
echo ""
echo "   To use: restart continuonbrain-startup.service"

