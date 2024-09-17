from typing import Dict

from fastapi import APIRouter, status

from app.schemas.text2motion import TextPrompt
from app.services.motion2mp4 import amass_motion2mp4_service
from app.services.text2motion import text2motion_generation_service

text2motion_router = APIRouter(prefix="")


@text2motion_router.post(
    "/text2motion",
    status_code=status.HTTP_200_OK,
    description="Generate stickman motion from a text prompt.",
    tags=["Text2Motion"],
)
async def text2motion_generation(request: TextPrompt) -> Dict[str, str]:
    motion_xyz, SMPL_params, text_embedding = text2motion_generation_service.infer(
        request
    )
    vid_name = f"{request.user_prompt}.mp4"
    amass_motion2mp4_service.create_visualization(
        motion_xyz, vid_name
    )
    return {"video_name": vid_name}
