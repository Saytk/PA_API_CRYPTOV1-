"""
API complète pour Crypto Forecast ML
Fournit des endpoints pour:
- Prédictions (actuelles et historiques)
- Informations sur les modèles
- Données historiques
- Authentification (repris de crypto_signals)
"""

from fastapi import FastAPI, Query, HTTPException, Depends, status, BackgroundTasks, Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path as PathLib
import pandas as pd
import logging
import jwt
from jwt.exceptions import PyJWTError

# Import project modules
from PA_ML.crypto_forecast_ml.crypto_signals_compat import (
    predict, get_historical_predictions, get_available_models,
    get_model_info, get_historical_data, load_last_candle
)
from PA_ML.crypto_forecast_ml.training.train_model import train_direction_model_with_timerange

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = PathLib(__file__).parent / "models"
RESULTS_DIR = PathLib(__file__).parent / "results"
SECRET_KEY = "YOUR_SECRET_KEY_HERE"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Forecast ML API",
    description="API complète pour accéder aux prédictions, modèles et données de Crypto Forecast ML",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ========================================================================= #
# Pydantic Models for Request/Response
# ========================================================================= #

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class PredictionRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
    use_incomplete_candle: bool = Field(True, description="Whether to use the incomplete current candle")

class BacktestRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
    n_folds: int = Field(5, description="Number of folds for walk-forward validation")

class TrainRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
    days: int = Field(30, description="Number of days of data to use for training")

class HistoricalDataRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
    days: int = Field(7, description="Number of days of historical data to retrieve")
    interval: str = Field("1m", description="Data interval (1m, 1h)")

# ========================================================================= #
# Authentication Functions
# ========================================================================= #

# This is a mock user database - in production, use a real database
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    # In production, use proper password hashing (e.g., bcrypt)
    return plain_password == "secret" and hashed_password == "fakehashedsecret"

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ========================================================================= #
# API Endpoints
# ========================================================================= #

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
def read_root():
    """API root endpoint with basic information"""
    return {
        "name": "Crypto Forecast ML API",
        "version": "1.0.0",
        "description": "API complète pour accéder aux prédictions, modèles et données de Crypto Forecast ML",
        "endpoints": [
            "/predict", "/predict/historical", "/models", "/models/{symbol}",
            "/backtest", "/data"
        ]
    }

@app.get("/predict")
def predict_latest(symbol: str = Query("BTCUSDT", description="Ex: BTCUSDT ou ETHUSDT"),
                  use_incomplete_candle: bool = Query(True, description="Utiliser la bougie en cours")):
    """
    Retourne la probabilité que la prochaine bougie minute ferme plus haut,
    ainsi qu'un signal discret LONG / FLAT / SHORT.
    """
    return predict(symbol, use_incomplete_candle)

@app.post("/predict/custom")
def predict_custom(request: PredictionRequest):
    """
    Endpoint personnalisé pour les prédictions avec options avancées
    """
    return predict(request.symbol, request.use_incomplete_candle)

@app.get("/predict/historical/{symbol}")
def get_historical_predictions_endpoint(
    symbol: str = Path(..., description="Trading pair symbol"),
    days: int = Query(7, description="Number of days of historical data")
):
    """
    Retourne les prédictions historiques pour un symbole donné
    """
    return get_historical_predictions(symbol, days)

@app.get("/models")
def get_models():
    """
    Retourne la liste des modèles disponibles
    """
    models = get_available_models()
    return {
        "available_models": models,
        "count": len(models)
    }

@app.get("/models/{symbol}")
def get_model_info_endpoint(symbol: str):
    """
    Retourne les informations détaillées sur un modèle spécifique
    """
    return get_model_info(symbol)

@app.post("/train")
async def train_model(
    request: TrainRequest, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """
    Déclenche l'entraînement de nouveaux modèles (tâche en arrière-plan)
    """
    # Start training in background for all horizons
    def train_all_horizons(symbol, days):
        for horizon in range(1, 6):
            model_file = MODEL_DIR / f"xgb_direction_h{horizon}_{symbol}.json"
            train_direction_model_with_timerange(
                symbol=symbol,
                days=days,
                output_path=str(model_file)
            )
            logger.info(f"Trained model for {symbol} with horizon {horizon}")

    background_tasks.add_task(train_all_horizons, request.symbol, request.days)

    return {
        "status": "training_started",
        "symbol": request.symbol,
        "message": f"Training started for {request.symbol} with 5 horizons. This may take several minutes."
    }

@app.get("/data/{symbol}")
def get_historical_data_endpoint(
    symbol: str,
    days: int  = Query(7,  description="Number of days of historical data"),
    interval: str = Query("1m", description="Data interval (1m, 1h)"),
    raw: bool = Query(False, description="Return plain array instead of wrapped object")
):
    """
    Return historical candle data.
    If ?raw=true, the response is a plain array of candles.
    """
    return get_historical_data(symbol, days, interval, raw)

@app.get("/data/{symbol}/last_candle")
async def get_last_candle(
    symbol: str = Path(..., description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
):
    """
    Returns only the last (most recent) candle for a given symbol
    """
    try:
        candle = load_last_candle(symbol)
        if not candle:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        return {
            "symbol": symbol,
            "timestamp": candle["timestamp_utc"],
            "candle": candle
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load last candle: {str(e)}")
