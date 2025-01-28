#!/bin/bash

# Input file
input_scp="/DKUdata/tangbl/data/urgent2025_challenge/data/speech_train_track1/gen/spk2fs2scp.scp"
base_path="/DKUdata/tangbl/data/urgent2025_challenge"

# Output directory for resampled audio
output_dir="/DKUdata/tangbl/data/urgent2025_challenge/resampled/speech"
mkdir -p "$output_dir"

# Function to resample audio
resample_audio() {
    uid="$1"
    fr="$2"
    input_path="$3"
    output_path="$output_dir/$input_path"

    mkdir -p "$(dirname "$output_path")"
    
    
    # Resample to 16kHz (adjust -ar value as needed)
    ffmpeg -i "$base_path/$input_path" -ar 16000 "$output_path" -y
}

# Export the function to make it available to xargs
export -f resample_audio
export output_dir
export base_path

# Use xargs to run resample_audio in parallel
cat "$input_scp" | xargs -n 3 -P 8 -I {} bash -c 'resample_audio "$@"' _ {}