#!/bin/bash

export MOTIONCLIP_DECODER_PATH=/Users/klabs/Documents/projects/text2motion_app/app/resources/model_state_dicts/MotionCLIP_Decoder_state_dict.pt
export CLIP_TEXT_ENCODER_PATH=/Users/klabs/Documents/projects/text2motion_app/app/resources/model_state_dicts/CLIP_ViT_B32_state_dict_no_ViT.pt
export SMPL_MODEL_PATH=/Users/klabs/Documents/projects/text2motion_app/app/resources/SMPL_models/SMPL_NEUTRAL.pkl
export JOINT_REGRESSOR_TRAIN_EXTRA=/Users/klabs/Documents/projects/text2motion_app/app/resources/SMPL_models/J_regressor_extra.npy
export VIDEO_OUPUT_DIR=/Users/klabs/Downloads/app_vids

gunicorn 'app.main:app' \
    --bind ${ADDRESS:-0.0.0.0}:${PORT:-8000} \
    --workers ${WORKERS:-1} \
    --worker-class uvicorn.workers.UvicornWorker \
    --log-level info \
    --access-logfile="-"