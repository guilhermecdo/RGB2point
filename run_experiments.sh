#!/bin/bash

# Define the array of the 5 dataset names exactly as they appear in your data_ready folder
DATASETS=("original" "blur" "gauss_blur" "denoise" "all_combined")

# Loop through each dataset
for DATASET in "${DATASETS[@]}"
do
    echo "================================================================="
    echo "🚀 STARTING EXPERIMENT: $DATASET"
    echo "================================================================="
    
    # Call your Python training script with the dynamic arguments
    # You can easily adjust the batch size and epochs here for all 5 runs
    python train.py \
        --experiment "$DATASET" \
        --data_root "data" \
        --comment "PINN-CROSMODAL-POSE" \
        --epochs 1000 \
        --batch_size 512
        
    echo "✅ FINISHED EXPERIMENT: $DATASET"
    echo "-----------------------------------------------------------------"
done

echo "🎉 ALL 5 DATASETS HAVE COMPLETED TRAINING!"