import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .core.config import settings
from .core.database import init_db
from .core.logging import configure_logging
from .services.inference import InferenceService

configure_logging()
logger = logging.getLogger(__name__)
inference_service = InferenceService(settings.model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        inference_service.load()
    except FileNotFoundError:
        logger.exception(
            "Model could not be loaded; API will remain available "
            "for health diagnostics"
        )
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error for %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
