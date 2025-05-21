#!/bin/bash

set -e

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

input_dir="$1"
output_dir="${input_dir}_cleaned"

mkdir -p "$output_dir"

shopt -s nullglob
for file in "$input_dir"/*.txt; do
  base=$(basename "$file")
  echo "Processing $base"
  awk '/^\*\*\* START OF THE PROJECT GUTENBERG EBOOK/{flag=1; next} flag' "$file" > "$output_dir/$base"
done
