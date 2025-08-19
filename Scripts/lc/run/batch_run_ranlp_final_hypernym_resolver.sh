#!/bin/bash

models=(
    "llama3.1:8b-instruct-q4_K_M"
    "phi4:14b-q4_K_M"
    "mistral:7b-instruct-v0.3-q4_K_M"
    "gemma3:12b-it-q4_K_M"
)

temps=(0.7)

shots=(0 1 5)

script="$SN3_ROOT/Scripts/lc/run/full_run_ranlp_final_hypernym_resolver.py"

# Number of parallel jobs allowed
max_jobs=1
running_jobs=0

for shot in "${shots[@]}"; do
    for temp in "${temps[@]}"; do
        for model in "${models[@]}"; do
            now=$(date +"%Y%m%d-%H%M%S")
            log_file="$SN3_ROOT/Runs/$model-$temp-$shot-$now.log"

            # Run the command in the background
            echo "Started: $model-$temp-$shot-$now"
            python3 "$script" --model="$model" --temperature="$temp" --examples="$shot" --retries=3 --ref="$now" > "$log_file" 2>&1
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
done

# Wait for remaining jobs to finish
wait