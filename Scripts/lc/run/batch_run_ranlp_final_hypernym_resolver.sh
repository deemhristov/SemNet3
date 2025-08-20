#!/bin/bash

models=(
    "llama3.1:8b-instruct-q4_K_M"
    "phi4:14b-q4_K_M"
    "gemma3:4b-it-q4_K_M"
    "mistral:7b-instruct-v0.3-q4_K_M"
)

temps=(0.7)

shots=(0 1 5)

export SN3_ROOT="/home/dcl/Projects/SemNet3"
script="$SN3_ROOT/Scripts/lc/run/full_run_ranlp_final_hypernym_resolver.py"

# Number of parallel jobs allowed
max_jobs=1
running_jobs=0

for model in "${models[@]}"; do
    for temp in "${temps[@]}"; do
        for shot in "${shots[@]}"; do
            now=$(date +"%Y%m%d-%H%M%S")
            json_file="$SN3_ROOT/Runs/$model-$temp-$shot-$now.json"
            out_file="$SN3_ROOT/Runs/$model-$temp-$shot-$now.out"
            err_file="$SN3_ROOT/Runs/$model-$temp-$shot-$now.err"

            # Run the command in the background
            echo "Started: $model-$temp-$shot-$now"
            python3 "$script" --model="$model" --temperature="$temp" --examples="$shot" --retries=3 --output="$json_file" > >(tee "$out_file") 2> "$err_file"
            echo "Finished: $model-$temp-$shot-$now"
            # ((running_jobs++))

            # If max_jobs reached, wait for any to finish
            # if (( running_jobs >= max_jobs )); then
            #     wait -n
            #     echo "Finished"
            #     ((running_jobs--))
            # fi
        done
    done
    ollama stop "$model" || true
done

# Wait for remaining jobs to finish
wait