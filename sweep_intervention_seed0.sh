#!/bin/bash
set -e

# Each entry: class_name | meaningful_find | meaningful_replace | control_find | control_replace
declare -a SWEEP=(
  "Glioma|glioma|schwannoma|showing|displaying"
  "Meningioma|meningioma|glioma|consistent|associated"
  "NORMAL|normal|abnormal|brain|cerebral"
  "Neurocitoma|neurocytoma|schwannoma|Intraventricular|Extra-axial"
  "Outros Tipos de Lesões|lesions|tumors|intracranial|external"
  "Schwannoma|schwannoma|glioma|Extra-axial|Intra-axial"
)

for entry in "${SWEEP[@]}"; do
  IFS='|' read -r cls m_find m_repl c_find c_repl <<< "$entry"
  out_name="${cls// /_}"
  echo "==== $cls: '$m_find' -> '$m_repl' (control: '$c_find' -> '$c_repl') ===="
  python run_intervention.py \
    --checkpoint results/fusion_seed0/best.pt \
    --brainiac_weights weights/brainiac/BrainIAC.ckpt \
    --data_root data \
    --batch_size 16 \
    --class_name "$cls" \
    --find "$m_find" \
    --replace "$m_repl" \
    --control_find "$c_find" \
    --control_replace "$c_repl" \
    --out "results/fusion_seed0/intervention_${out_name}.json"
done
