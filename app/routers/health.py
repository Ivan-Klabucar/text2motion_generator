from fastapi import APIRouter, status
from typing import Dict

health_router = APIRouter(prefix="")


@health_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    description="Liveness Text2Motion API health check",
    tags=["Text2Motion Generation"],
)
def health_check() -> Dict[str, str]:
    return {"health": "UP"}
