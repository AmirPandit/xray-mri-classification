# src/services/database_service.py
import sqlite3
from pathlib import Path
import logging
from typing import Optional, List, Dict, Any
import json
from datetime import datetime

from models.user_model import User, Base
from src.config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database operations."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.BASE_DIR / "database.db"
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    full_name TEXT,
                    hashed_password TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create predictions table for history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    image_filename TEXT,
                    prediction TEXT,
                    confidence REAL,
                    all_predictions TEXT,
                    model_version TEXT,
                    inference_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create refresh tokens table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    is_revoked BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_user(self, user_data: dict) -> Optional[dict]:
        """Create a new user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            user_id = user_data.get('id')
            cursor.execute('''
                INSERT INTO users (id, email, username, full_name, hashed_password, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                user_data['email'],
                user_data['username'],
                user_data.get('full_name'),
                user_data['hashed_password'],
                user_data.get('is_active', True)
            ))
            
            conn.commit()
            
            # Return created user
            return self.get_user_by_id(user_id)
            
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed (integrity error): {e}")
            return None
        except sqlite3.Error as e:
            logger.error(f"User creation failed: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM users WHERE email = ?
            ''', (email,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_user_dict(row)
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error getting user by email: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Get user by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM users WHERE id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_user_dict(row)
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get user by username."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM users WHERE username = ?
            ''', (username,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_user_dict(row)
            return None
            
        except sqlite3.Error as e:
            logger.error(f"Error getting user by username: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def _row_to_user_dict(self, row) -> dict:
        """Convert database row to user dictionary."""
        return {
            "id": row[0],
            "email": row[1],
            "username": row[2],
            "full_name": row[3],
            "hashed_password": row[4],
            "is_active": bool(row[5]),
            "created_at": row[6],
            "updated_at": row[7]
        }
    
    def save_prediction(self, prediction_data: dict) -> bool:
        """Save prediction to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (id, user_id, image_filename, prediction, confidence, 
                                       all_predictions, model_version, inference_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data.get('id'),
                prediction_data.get('user_id'),
                prediction_data.get('image_filename'),
                prediction_data.get('prediction'),
                prediction_data.get('confidence'),
                json.dumps(prediction_data.get('all_predictions', {})),
                prediction_data.get('model_version'),
                prediction_data.get('inference_time')
            ))
            
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error saving prediction: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_user_predictions(self, user_id: str, limit: int = 10) -> List[dict]:
        """Get predictions for a specific user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            rows = cursor.fetchall()
            predictions = []
            
            for row in rows:
                predictions.append({
                    "id": row[0],
                    "user_id": row[1],
                    "image_filename": row[2],
                    "prediction": row[3],
                    "confidence": row[4],
                    "all_predictions": json.loads(row[5]) if row[5] else {},
                    "model_version": row[6],
                    "inference_time": row[7],
                    "created_at": row[8]
                })
            
            return predictions
            
        except sqlite3.Error as e:
            logger.error(f"Error getting user predictions: {e}")
            return []
        finally:
            if conn:
                conn.close()

# Global database service instance
database_service = DatabaseService()