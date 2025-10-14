#!/bin/bash
# run script : bash run_spectral_wise_depth.sh

# Model: Midas_dinov2
CKPT1="./checkpoints/Midas_dinov2_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"
CKPT2="./checkpoints/Midas_thermal_dinov2_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"

SAVE_THR="./result_icra26_new/Multi_depth/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_dinov2_pretrained_frozen_backbone.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_multi_monodepth.py --config ${CONFIG} --ckpt_paths ${CKPT1} ${CKPT2} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ}
done