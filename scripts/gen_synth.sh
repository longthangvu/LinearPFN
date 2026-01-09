#!/bin/bash

while read -r dataset_id; do
    ds_dir="series_bank/$dataset_id"

    echo "$(date +"%T")"
    echo "_____________ Generating to: $ds_dir _____________"
    python synthetic_prior/main.py --dataset_id $dataset_id

    echo "Done: $(date +"%T")"
done < jobs_synth.txt 