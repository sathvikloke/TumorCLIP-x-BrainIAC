#!/usr/bin/env bash
# Download the Kaggle 17-class brain tumor MRI dataset and run the 17->6
# superclass consolidation. Requires the Kaggle CLI configured with an API key.
#
# Usage:
#   bash scripts/download_data.sh
set -euo pipefail

mkdir -p data/raw data/processed

if ! command -v kaggle &>/dev/null; then
  echo "ERROR: kaggle CLI not found. Install with: pip install kaggle"
  echo "Then place your API token at ~/.kaggle/kaggle.json (chmod 600)."
  exit 1
fi

echo "Downloading dataset..."
kaggle datasets download -d fernando2rad/brain-tumor-mri-images-17-classes -p data/raw/
unzip -o data/raw/brain-tumor-mri-images-17-classes.zip -d data/raw/
rm data/raw/brain-tumor-mri-images-17-classes.zip

echo "Consolidating classes -> 6 superclasses..."
# consolidate_classes.py auto-descends if data/raw has a single subdir wrapper,
# so we just point it at data/raw and let it figure things out.
python scripts/consolidate_classes.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --test_frac 0.35 \
  --seed 42

echo "Done. Processed data at: data/processed/{train,test}/"
echo "If any classes fell through to 'Other Lesions', inspect data/raw/ and"
echo "edit MAPPING in scripts/consolidate_classes.py, then rerun the script."
