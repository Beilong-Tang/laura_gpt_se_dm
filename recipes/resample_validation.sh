#!/bin/bash

# Input file
base_path="/DKUdata/tangbl/data/urgent2025_challenge"

# Output directory for resampled audio
target_sr=16000

# Function to resample audio
resample_audio() {
    uid="$1"
    fr="$2"
    input_path="$3"
    output_path="$output_dir/$input_path"

    mkdir -p "$(dirname "$output_path")"
    
    # Resample to 16kHz (adjust -ar value as needed)
    ffmpeg -i "$base_path/$input_path" -map_channel 0.0.0 -ar $target_sr "$output_path" -y
}

# Export the function to make it available to xargs
export -f resample_audio
export base_path
export target_sr

# Use xargs to run resample_audio in parallel

## Inference clean speech
output_dir="/DKUdata/tangbl/data/urgent2025_challenge/resampled/validation/clean_$target_sr"
input_scp="/DKUdata/tangbl/data/urgent2025_challenge/data/validation/spk1.scp"
export output_dir

cat "$input_scp" | xargs -n 3 -P 4 bash -c 'resample_audio "$@"' _

## Inference nosiy speech
output_dir="/DKUdata/tangbl/data/urgent2025_challenge/resampled/validation/noisy_$target_sr"
input_scp="/DKUdata/tangbl/data/urgent2025_challenge/data/validation/wav.scp"
export output_dir

cat "$input_scp" | xargs -n 3 -P 4 bash -c 'resample_audio "$@"' _


echo "Done"

