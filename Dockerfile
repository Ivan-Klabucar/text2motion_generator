FROM python:3.8-slim

# Install FFmpeg
RUN apt-get update && \
    apt-get install -y \
        ffmpeg \
        git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p /src/app/videos

ENV MOTIONCLIP_DECODER_PATH=/src/app/resources/model_state_dicts/MotionCLIP_Decoder_state_dict.pt
ENV CLIP_TEXT_ENCODER_PATH=/src/app/resources/model_state_dicts/CLIP_ViT_B32_state_dict_no_ViT.pt
ENV SMPL_MODEL_PATH=/src/app/resources/SMPL_models/SMPL_NEUTRAL.pkl
ENV JOINT_REGRESSOR_TRAIN_EXTRA=/src/app/resources/SMPL_models/J_regressor_extra.npy
ENV VIDEO_OUPUT_DIR=/src/app/videos
ENV S3_BUCKET_NAME=text2motion-backups

EXPOSE 8000

RUN chmod +x ./start.sh

CMD ["./start.sh"]