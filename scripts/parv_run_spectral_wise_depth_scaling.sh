#!/bin/bash
# run script : bash run_spectral_wise_depth.sh

# Model: Midas_thermal_dinov2_boson
CKPT="./checkpoints/Midas_thermal_dinov2_boson_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"

SAVE_THR="./result_icra26/scaling/Midas_boson/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_dinov2_pretrained_frozen_backbone.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# Model: Midas_thermal_dinov2_boson_vivid
CKPT="./checkpoints/Midas_thermal_dinov2_boson_vivid_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"

SAVE_THR="./result_icra26/scaling/Midas_boson_vivid/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_dinov2_pretrained_frozen_backbone.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# Model: Midas_thermal_dinov2_boson_vivid_freiburg
CKPT="./checkpoints/Midas_thermal_dinov2_boson_vivid_freiburg_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"

SAVE_THR="./result_icra26/scaling/Midas_boson_vivid_freiburg/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_dinov2_pretrained_frozen_backbone.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# Model: Midas_thermal_dinov2_boson_vivid_freiburg_sthereo
CKPT="./checkpoints/Midas_thermal_dinov2_boson_vivid_sthereo_freiburg_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"

SAVE_THR="./result_icra26/scaling/Midas_boson_vivid_freiburg_sthereo/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_dinov2_pretrained_frozen_backbone.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# Model: Midas_thermal_dinov2_boson_vivid_freiburg_sthereo_tartanrgbt
CKPT="./checkpoints/Midas_thermal_dinov2_pretrained_frozen_backbone/ckpt_epoch=28_step=145000.ckpt"

SAVE_THR="./result_icra26/scaling/Midas_thermal_dinov2_boson_vivid_freiburg_sthereo_tartanrgbt/thr"

mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/Midas_dinov2_pretrained_frozen_backbone.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python3 test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

