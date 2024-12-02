#!/bin/bash

BASE_PATH="assignments/05/3_2"
output_dir="$BASE_PATH"
time_start=$(date +%s)
python translate_beam_constant.py \
    --data data/en-fr/prepared/ \
    --dicts data/en-fr/prepared \
    --checkpoint-path "assignments/03/baseline/checkpoints/checkpoint_best.pt" \
    --output "$output_dir/translations.txt" \
    --batch-size 16
time_end=$(date +%s)
real_time=$((time_end-time_start))
bash scripts/postprocess.sh \
"$output_dir/translations.txt" \
"$output_dir/translations.p.txt" \
en
cat "$output_dir/translations.p.txt" | sacrebleu data/en-fr/raw/test.en > "$output_dir/bleu.txt"
