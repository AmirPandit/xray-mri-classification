# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from src.config.logging_config import setup_logging
from src.config.settings import settings
from src.services.prediction_service import prediction_service
from src.services.database_service import database_service

# Import routers
from src.api.auth_routes import router as auth_router
from src.api.routes import router as api_router

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown."""
    logger.info("Starting X-ray/MRI Classification API...")

    # Initialize database
    try:
        database_service._init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Load model
    model_loaded = prediction_service.load_model()
    if not model_loaded:
        logger.error("Failed to load model on startup")
    else:
        logger.info("Model loaded successfully")

    yield  # Run the application

    # Shutdown cleanup
    logger.info("Shutting down X-ray/MRI Classification API...")

# Create FastAPI application
app = FastAPI(
    title="X-ray/MRI Classification API",
    description="A deep learning API for classifying medical images with authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(api_router, prefix="/api/v1", tags=["predictions"])

@app.get("/")
async def root():
    return {
        "message": "X-ray/MRI Classification API with Authentication",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": prediction_service.model_loaded,
        "authentication": "enabled"
    }

@app.get("/info")
async def api_info():
    return {
        "name": "X-ray/MRI Classification API",
        "version": "1.0.0",
        "description": "Deep learning API for medical image classification with authentication",
        "authentication": True,
        "endpoints": {
            "auth": {
                "register": "/api/v1/auth/register",
                "login": "/api/v1/auth/login", 
                "logout": "/api/v1/auth/logout",
                "me": "/api/v1/auth/me"
            },
            "predictions": {
                "predict": "/api/v1/predict",
                "predict-batch": "/api/v1/predict-batch",
                "history": "/api/v1/history"
            }
        }
    }
