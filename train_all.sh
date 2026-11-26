#!/usr/bin/env bash
# Train one SDXL LoRA adapter per style.
# Produces 5 .safetensors files under ../outputs/lora/<style>/.
# Requires kohya-ss/sd-scripts clone; set SD_SCRIPTS to its path.

set -euo pipefail

SD_SCRIPTS="${SD_SCRIPTS:-$HOME/sd-scripts}"
STYLES=(ink_wash ukiyo_e cyberpunk baroque minimalist)

for style in "${STYLES[@]}"; do
  echo "=== Training LoRA for style: ${style} ==="
  accelerate launch --num_processes 1 \
    "${SD_SCRIPTS}/sdxl_train_network.py" \
    --config_file "configs/${style}.toml"
done

echo "All 5 style LoRAs trained. Outputs in ../outputs/lora/"
