#!/bin/bash

# Collect all JSON filenames into an array
files=(/home/dcl/Projects/SemNet3/Runs/Final\ 2/*.json)

# Get the number of files
n=${#files[@]}

# Empty tsv string with header columns <empty>\t<file1>\t<file2>
tsv_string=""
for ((i = 0; i < n; i++)); do
    tsv_string+="\t$(basename "${files[i]}")"
done

# Iterate over all unique pairs (i < j)
for ((i = 0; i < n; i++)); do
    tsv_string+="\n$(basename "${files[i]}")"
    for ((j = 0; j < n; j++)); do
        a=$(python3 /home/dcl/Projects/SemNet3/Scripts/measure_ab.py "${files[i]}" "${files[j]}")
        tsv_string+="\t$a"
    done
done

echo -e "$tsv_string" > /home/dcl/Projects/SemNet3/Runs/Final\ 2/measure_ab.tsv