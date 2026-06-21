#!/bin/bash
# run script : bash run_spectral_wise_depth.sh

# Model: Midas_dinov2
CKPT="./checkpoints/Midas_dinov2_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"

SAVE_THR="./result_icra26/Midas_dinov2/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_dinov2_pretrained_frozen_backbone.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# Model: Midas_anythermal
CKPT="./checkpoints/Midas_anythermal_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"

SAVE_THR="./result_icra26/Midas_anythermal/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_dinov2_pretrained_frozen_backbone.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# Model: Midas_small
CKPT="./checkpoints/Midas_small/ckpt_epoch=26_step=135000.ckpt"

SAVE_THR="./result_icra26/Midas_small/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_small.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done