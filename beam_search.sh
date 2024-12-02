#!/bin/bash

BASE_PATH="assignments/05"
mkdir -p "$BASE_PATH/original"
timing_log="$BASE_PATH/beam_decoding_times.csv"
echo "Beam Size,Real Time" > "$timing_log"

for beam_size in {1..25}; do
    printf -v padded_size "%02d" $beam_size
    output_dir="$BASE_PATH/original/beam_${padded_size}"
    mkdir -p "$output_dir"

    echo "Running beam search with beam size $beam_size"

    time_start=$(date +%s)
    python translate_beam.py \
        --beam-size "$beam_size" \
        --data data/en-fr/prepared/ \
        --dicts data/en-fr/prepared \
        --checkpoint-path "assignments/03/baseline/checkpoints/checkpoint_best.pt" \
        --output "$output_dir/translations.txt" \
        --batch-size 16
    time_end=$(date +%s)
    real_time=$((time_end-time_start))
    echo "$beam_size, $real_time" >> "$timing_log"
    echo "$real_time" >> "$output_dir/beam_decoding_time.log"
    echo "Done translating with checkpoint $checkpoint"

    echo "Postprocessing translations"
    bash scripts/postprocess.sh \
        "$output_dir/translations.txt" \
        "$output_dir/translations.p.txt" \
        en
    echo "Done postprocessing translations"

    echo "Calculating BLEU score for beam size $beam_size:"
    cat "$output_dir/translations.p.txt" | sacrebleu data/en-fr/raw/test.en > "$output_dir/bleu.txt"
    
        
    if [ $? -ne 0 ]; then
        echo "Error with beam size $padded_size" >> "$BASE_PATH/errors.log"
    fi
done