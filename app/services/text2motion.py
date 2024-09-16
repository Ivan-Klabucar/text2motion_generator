import os

import clip
import torch

from app.schemas.text2motion import TextPrompt

from .models.CLIP_TextEncoder import CLIP_TextEncoder
from .models.MotionCLIP import Decoder_TRANSFORMER as MotionCLIP_Decoder
from .models.MotionCLIP import rot2xyz
from .models.trajectory_correction import correct_AMASS_sequence


class Text2Motion_GenerationPipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.tokenizer = clip.tokenize
        self.text_encoder = CLIP_TextEncoder(
            embed_dim=512,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
        self.motion_decoder = MotionCLIP_Decoder(
            modeltype="cvae",
            njoints=25,
            nfeats=6,
            num_frames=60,
            num_classes=0,
            translation=True,
            pose_rep="rot6d",
            glob=True,
            glob_rot=[3.141592653589793, 0, 0],
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
        )

        self.motion_decoder.load_state_dict(
            torch.load(os.getenv("MOTIONCLIP_DECODER_PATH"), weights_only=True)
        )
        self.motion_decoder.eval()

        self.text_encoder.load_state_dict(
            torch.load(os.getenv("CLIP_TEXT_ENCODER_PATH"), weights_only=True)
        )
        self.text_encoder.eval()

    @torch.no_grad()
    def infer(self, request: TextPrompt) -> dict:
        with torch.no_grad():
            text_inputs = self.tokenizer(request.user_prompt).to(self.device)
            text_embedding = self.text_encoder(text_inputs)  # Shape (1,512)

            motion_dec_input = {
                "clip_text_emb": text_embedding,
                "y": None,
                "mask": torch.ones(1, 60, dtype=torch.bool),
                "lengths": None,
            }
            SMPL_motion = self.motion_decoder(motion_dec_input)
            motion_xyz = rot2xyz(SMPL_motion["txt_output"], SMPL_motion["mask"])
            motion_xyz = motion_xyz.permute(
                0, 3, 1, 2
            )  # Shape (1, 60, 24, 3) or (num_samples, frames, joints, coordinate axis)
            motion_xyz = correct_AMASS_sequence(motion_xyz[0].numpy())
            return motion_xyz, SMPL_motion["txt_output"], text_embedding


text2motion_generation_service = Text2Motion_GenerationPipeline()
