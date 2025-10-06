from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, status
import logging
import time
import uuid
import base64
from typing import List

from src.api.schemas import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, PredictionHistory, HealthResponse
)
from src.services.prediction_service import prediction_service
from src.services.auth_service import get_current_user
from src.services.database_service import database_service
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.model_loaded,
        model_version=prediction_service.model_version if prediction_service.model_loaded else None,
        timestamp=time.time()
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict endpoint for single image (with user context)."""
    try:
        if not prediction_service.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        # Make prediction
        result = prediction_service.predict(request.image_data, request.include_heatmap)

        # Save prediction to database
        prediction_id = str(uuid.uuid4())
        prediction_data = {
            "id": prediction_id,
            "user_id": current_user["id"],
            "image_filename": f"prediction_{prediction_id}.jpg",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "all_predictions": result.get("all_predictions"),
            "model_version": result.get("model_version"),
            "inference_time": result.get("inference_time")
        }
        database_service.save_prediction(prediction_data)
        logger.info(f"Prediction saved for user: {current_user['email']}")

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Batch prediction endpoint for multiple images."""
    try:
        if not prediction_service.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        start_time = time.time()
        predictions = prediction_service.predict_batch(request.images, request.include_heatmaps)
        total_time = time.time() - start_time

        for pred in predictions:
            if "error" not in pred:
                prediction_id = str(uuid.uuid4())
                database_service.save_prediction({
                    "id": prediction_id,
                    "user_id": current_user["id"],
                    "image_filename": f"prediction_{prediction_id}.jpg",
                    "prediction": pred["prediction"],
                    "confidence": pred["confidence"],
                    "all_predictions": pred.get("all_predictions"),
                    "model_version": pred.get("model_version"),
                    "inference_time": pred.get("inference_time")
                })

        return BatchPredictionResponse(
            predictions=[PredictionResponse(**pred) for pred in predictions],
            total_time=total_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.get("/history", response_model=List[PredictionHistory])
async def get_prediction_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get prediction history for current user."""
    try:
        predictions = database_service.get_user_predictions(current_user["id"], limit)
        history = [
            PredictionHistory(
                id=pred["id"],
                timestamp=pred["created_at"],
                prediction=pred["prediction"],
                confidence=pred["confidence"],
                image_size="unknown",
                model_version=pred.get("model_version")
            )
            for pred in predictions
        ]
        return history

    except Exception as e:
        logger.error(f"History endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )

@router.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(
    file: UploadFile = File(...),
    include_heatmap: bool = True,
    current_user: dict = Depends(get_current_user)
):
    """Predict endpoint for file upload."""
    try:
        if not prediction_service.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )

        contents = await file.read()
        image_data = base64.b64encode(contents).decode('utf-8')

        result = prediction_service.predict(image_data, include_heatmap)

        prediction_id = str(uuid.uuid4())
        database_service.save_prediction({
            "id": prediction_id,
            "user_id": current_user["id"],
            "image_filename": f"prediction_{prediction_id}.jpg",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "all_predictions": result.get("all_predictions"),
            "model_version": result.get("model_version"),
            "inference_time": result.get("inference_time")
        })

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload prediction failed: {str(e)}"
        )
