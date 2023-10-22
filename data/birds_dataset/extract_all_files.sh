#!/bin/bash
mkdir -p images

root_directory="/home/katsiaryna/Projects/Generative models/data/birds_dataset/archive/"
train_directory="$root_directory/train/"
valid_directory="$root_directory/valid/"
test_directory="$root_directory/test/"
source_directories=(
   "$train_directory"
   "$valid_directory"
   "$test_directory"
)
destination_directory="/home/katsiaryna/Projects/Generative models/data/birds_dataset/images/"

for source_dir in "${source_directories[@]}"; do
   find "$source_dir" -type f -exec bash -c 'new_name="$2/$(basename "$(dirname "$1")")_$(basename "$1")"; mv "$1" "$new_name";' bash {} "$destination_directory" \;
done

