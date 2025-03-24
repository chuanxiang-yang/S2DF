#!/bin/bash
DIGS_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))
DATASET_PATH=$DIGS_DIR'/data/3dscene/input' # change to your dataset path
echo "If $DIGS_DIR is not the correct path for your DiGS repository, set it manually at the variable DIGS_DIR"
echo "If $DATASET_PATH is not the correct path for the NSP dataset, change the variable DATASET_PATH"

cd $DIGS_DIR/surface_reconstruction/ # To call python scripts correctly

LOGDIR='./log/3dscene' #change to your desired log directory
mkdir -p $LOGDIR
FILE=`basename "$0"`

LAYERS=4
DECODER_HIDDEN_DIM=256
NL='sine' # 'sine' | 'relu' | 'softplus'
INIT_TYPE='siren' #siren | geometric_sine | geometric_relu | mfgi

#LOSS_WEIGHTS=(100000000 1000000 8000000 0.0085)
LOSS_WEIGHTS=(100000000 1000000 8000000 0.006)

NPOINTS=15000
NITERATIONS=10000
neuron_type='linear'
GPU=1
LR=3e-4
CONF='../confs/3dscene.conf'

# For subset, use e.g.,
for FILENAME in $DATASET_PATH/*; do
      FILENAME="$(basename "$FILENAME")"
      echo $FILENAME
      SCAN_PATH=$DATASET_PATH
      python train_surface_reconstruction.py --logdir $LOGDIR/ --file_name $FILENAME --gpu_idx $GPU --n_iterations $NITERATIONS --n_points $NPOINTS --lr ${LR} --dataset_path $SCAN_PATH --decoder_n_hidden_layers $LAYERS --decoder_hidden_dim ${DECODER_HIDDEN_DIM} --init_type ${INIT_TYPE}  --neuron_type ${neuron_type} --nl ${NL}  --loss_weights ${LOSS_WEIGHTS[@]}
      python test_surface_reconstruction.py --logdir $LOGDIR --file_name $FILENAME --dataset_path $SCAN_PATH --gpu_idx $GPU --epoch_n $NITERATIONS --conf $CONF
done

