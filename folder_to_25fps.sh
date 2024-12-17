#!/bin/bash

# Check if the user provided a folder path as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/folder"
  exit 1
fi

# Assign the provided folder path to a variable
input_folder="$1"

# Verify that the folder exists
if [ ! -d "$input_folder" ]; then
  echo "Error: The specified folder does not exist: $input_folder"
  exit 1
fi

# Change directory to the provided folder
cd "$input_folder" || exit

# Loop through all .mp4 video files in the folder
for file in *.mp4; do
  # Skip if no .mp4 files are found
  [ -e "$file" ] || continue

  # Extract the filename without extension
  filename="${file%.*}"
  
  # Create a new output file with '_25fps' suffix
  output="${filename}_25fps.mp4"
  
  # Use ffmpeg to resample to 25 fps
  ffmpeg -i "$file" -r 25 -c:v libx264 -crf 23 -preset medium -c:a aac "$output"
done
