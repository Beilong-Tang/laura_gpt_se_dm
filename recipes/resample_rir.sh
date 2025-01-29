#!/bin/bash

# Input file
input_scp="/DKUdata/tangbl/data/urgent2025_challenge/data/rir_train.scp"
base_path="/DKUdata/tangbl/data/urgent2025_challenge"

# Output directory for resampled audio
target_sr=16000
output_dir="/DKUdata/tangbl/data/urgent2025_challenge/resampled/rir_$target_sr"
mkdir -p "$output_dir"

# Function to resample audio
resample_audio() {
    uid="$1"
    fr="$2"
    input_path="$3"
    output_path="$output_dir/$input_path"

    mkdir -p "$(dirname "$output_path")"
    
    # echo $uid
    # echo $fr
    # echo $input_path

    # Resample to 16kHz (adjust -ar value as needed)
    ffmpeg -i "$base_path/$input_path" -map_channel 0.0.0 -ar $target_sr "$output_path" -y
    # ffmpeg -i "$full_input_path" -map_channel 0.0.0 -ar 16000 "$output_path" -y
    # echo $output_path
    # echo "$base_path/$input_path"
}

# Export the function to make it available to xargs
export -f resample_audio
export output_dir
export base_path
export target_sr

# Use xargs to run resample_audio in parallel
cat "$input_scp" | xargs -n 3 -P 12 bash -c 'resample_audio "$@"' _

echo "Done"


