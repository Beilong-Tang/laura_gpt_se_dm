#!/bin/bash

python recipes/resample_scp.py --scp "/DKUdata/tangbl/data/urgent2025_challenge/data/speech_train_track1/gen/spk2fs2scp.scp" \
    --base_path "/DKUdata/tangbl/data/urgent2025_challenge" \
    --output_dir "/DKUdata/tangbl/data/urgent2025_challenge/resampled/speech" \
    --sample_rate 16000 \
    --num_proc 8