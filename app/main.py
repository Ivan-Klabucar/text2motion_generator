from fastapi import FastAPI

from app.routers.health import health_router
from app.routers.text2motion import text2motion_router

app = FastAPI(
    title="Text2Motion API",
    docs_url="/",
    debug=False,
)
app.include_router(health_router)
app.include_router(text2motion_router)
