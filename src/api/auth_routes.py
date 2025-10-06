from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from datetime import datetime

from src.api.schemas import (
    UserCreate, UserResponse, UserLogin, Token, 
    LogoutResponse, HealthResponse
)
from src.services.auth_service import auth_service, get_current_user
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user."""
    try:
        user = auth_service.register_user(user_data)
        logger.info(f"New user registered: {user['email']}")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=Token)
async def login(login_data: UserLogin):
    """Login user and return access token."""
    try:
        result = auth_service.login_user(login_data)
        logger.info(f"User logged in: {login_data.email}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/logout", response_model=LogoutResponse)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    current_user: dict = Depends(get_current_user)
):
    """Logout user."""
    try:
        token = credentials.credentials
        success = auth_service.logout_user(token)
        
        if success:
            logger.info(f"User logged out: {current_user['email']}")
            return LogoutResponse(
                message="Successfully logged out",
                success=True
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout failed"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    return current_user

@router.put("/me", response_model=UserResponse)
async def update_current_user_info(
    user_update: UserCreate,  # In real app, create UserUpdate schema
    current_user: dict = Depends(get_current_user)
):
    """Update current user information."""
    # This is a placeholder - implement proper user update logic
    # For now, just return current user
    logger.info(f"User update requested for: {current_user['email']}")
    return current_user

@router.get("/auth/health", response_model=HealthResponse)
async def auth_health_check():
    """Authentication health check."""
    return HealthResponse(
        status="healthy",
        service="authentication",
        timestamp=datetime.now().isoformat()
    )