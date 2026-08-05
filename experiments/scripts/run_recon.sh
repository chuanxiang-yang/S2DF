#!/bin/bash
DIGS_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))

DATASET_PATH='./data/Stanford/ground_truth'
for FILENAME in $DATASET_PATH/*; do
    FILENAME="$(basename "$FILENAME")"
    echo "File: ${FILENAME}"

    python train_sdf.py --mesh_path $DATASET_PATH/$FILENAME --experiment_name $FILENAME --logging_root './res/Stanford' --num_steps 100000 \
        --lr 0.0001 --hidden_size 256 --hidden_layers 4  --octree_depth 8 \
        --w0 30 --fbs 1.0 \
        --gpu 0

    python test_sdf.py --mesh_path $DATASET_PATH/$FILENAME --experiment_name $FILENAME --logging_root './res/Stanford' --num_steps 100000 \
        --lr 0.0001 --hidden_size 256 --hidden_layers 4  \
        --w0 30 --fbs 1.0 \
        --gpu 0
done






