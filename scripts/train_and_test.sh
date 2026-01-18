#!/bin/bash

while read -r model model_id ckpts_root; do
    out_dir="$ckpts_root/$model/$model_id"
    train_log_file="$out_dir/train.log"
    test_log_file="$out_dir/tslib_tests.log"

    for seed in 42 67 69; do

        echo "$(date +"%T")"
        echo "_____________ Training: $model-$model_id-seed:$seed _____________"
        python training/train.py --model $model --model_id $model_id --ckpts_root $ckpts_root --seed $seed >> $train_log_file

        echo "$(date +"%T")"
        echo "_____________ Testing: $model-$model_id-seed:$seed _____________"
        bash scripts/test_tslib.sh --model $model --model_id $model_id --ckpts_root $ckpts_root --seed $seed >> $test_log_file

        echo "Done: $(date +"%T")"
    done
done < jobs.txt 