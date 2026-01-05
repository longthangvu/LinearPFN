#!/bin/bash

while read -r model model_id ckpts_root; do
    out_dir="$ckpts_root/$model/$model_id"
    train_log_file="$out_dir/train.log"
    test_log_file="$out_dir/tslib_tests.log"

    echo "$(date +"%T")"
    echo "_____________ Training: $model-$model_id _____________"
    python training/train.py --model $model --model_id $model_id --ckpts_root $ckpts_root > $train_log_file

    echo "$(date +"%T")"
    echo "_____________ Testing: $model-$model_id _____________"
    bash scripts/test_tslib.sh --model $model --model_id $model_id --ckpts_root $ckpts_root > $test_log_file

    echo "Done: $(date +"%T")"
done < jobs.txt 