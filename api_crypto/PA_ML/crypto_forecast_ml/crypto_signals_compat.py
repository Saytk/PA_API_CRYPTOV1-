"""
Compatibility module to replace crypto_signals functionality with crypto_forecast_ml.
This module provides the same API as crypto_signals but uses the XGBoost models from crypto_forecast_ml.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Import crypto_forecast_ml modules
from PA_ML.crypto_forecast_ml.data_loader import load_crypto_data, load_crypto_data_custom_range
from PA_ML.crypto_forecast_ml.features.feature_engineering import add_all_features
from PA_ML.crypto_forecast_ml.features.target_builder import build_targets

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global dictionaries to store models and metadata
MODELS = {}
METADATA = {}

# Flag to use the aggregated model instead of individual horizon models
USE_AGGREGATED_MODEL = True

def load_aggregated_model(symbol: str) -> bool:
    """
    Load the aggregated 5-minute model and its metadata.

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)

    Returns:
        bool: True if loading was successful, False otherwise
    """
    model_file = MODEL_DIR / f"xgb_direction_5min_{symbol}.json"
    # The metadata file path should match the one created in train_aggregated_model
    metadata_path = str(model_file).replace(".json", "_metadata.json")
    metadata_file = Path(metadata_path)

    # If model doesn't exist, train it
    if not model_file.exists():
        logger.warning(f"Aggregated model file for {symbol} not found. Training a new model.")
        from PA_ML.crypto_forecast_ml.training.train_model import train_aggregated_model

        # Train and save model
        train_aggregated_model(
            symbol=symbol,
            days=30,
            output_path=str(model_file)
        )

        if not model_file.exists():
            logger.error(f"Failed to create aggregated model file for {symbol}")
            return False

    # Load metadata if it exists
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                file_content = f.read().strip()
                if file_content:  # Check if file is not empty
                    metadata = json.loads(file_content)
                else:
                    logger.warning(f"Metadata file {metadata_file} is empty. Creating default metadata.")
                    raise ValueError("Empty metadata file")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error loading metadata file {metadata_file}: {str(e)}. Creating default metadata.")
            # Fall through to create default metadata
            metadata = None
    else:
        metadata = None

    # Create default metadata if it doesn't exist or couldn't be loaded
    if metadata is None:
        # Create default metadata
        metadata = {
            "symbol": symbol,
            "created_at": datetime.now().isoformat(),
            "model_type": "xgboost_5min",
            "features": [
                "open", "high", "low", "close", "volume", "quote_volume", "nb_trades",
                "sma_5", "sma_10", "ema_5", "ema_10", "rsi_14",
                "macd", "macd_signal", "macd_diff",
                "bb_upper", "bb_lower", "bb_width",
                "atr_14"
            ],
            "pred_thresh": 0.5,
            "description": "Modèle agrégé pour prédiction sur 5 minutes"
        }
        # Write default metadata to file
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    try:
        # Load the model
        model_key = f"{symbol}_5min"
        MODELS[model_key] = xgb.Booster()
        MODELS[model_key].load_model(str(model_file))

        # Store metadata
        METADATA[model_key] = metadata

        logger.info(f"Aggregated model loaded: {model_file}")
        return True
    except Exception as e:
        logger.error(f"Error loading aggregated model for {symbol}: {str(e)}")
        return False

def load_model(symbol: str, horizon: int = 1) -> bool:
    """
    Load a model and its metadata on demand.

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        horizon: Number of periods in the future to predict (1-5)

    Returns:
        bool: True if loading was successful, False otherwise
    """
    # If using aggregated model and this is the first horizon, load the aggregated model instead
    if USE_AGGREGATED_MODEL and horizon == 1:
        return load_aggregated_model(symbol)

    model_file = MODEL_DIR / f"xgb_direction_h{horizon}_{symbol}.json"
    metadata_file = MODEL_DIR / f"metadata_h{horizon}_{symbol}.json"

    # Load metadata if it exists
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                file_content = f.read().strip()
                if file_content:  # Check if file is not empty
                    metadata = json.loads(file_content)
                else:
                    logger.warning(f"Metadata file {metadata_file} is empty. Creating default metadata.")
                    raise ValueError("Empty metadata file")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error loading metadata file {metadata_file}: {str(e)}. Creating default metadata.")
            # Fall through to create default metadata
            metadata = None
    else:
        metadata = None

    # Create default metadata if it doesn't exist or couldn't be loaded
    if metadata is None:
        metadata = {
            "symbol": symbol,
            "horizon": horizon,
            "created_at": datetime.now().isoformat(),
            "model_type": "xgboost",
            "features": [
                "open", "high", "low", "close", "volume", "quote_volume", "nb_trades",
                "sma_5", "sma_10", "ema_5", "ema_10", "rsi_14",
                "macd", "macd_signal", "macd_diff",
                "bb_upper", "bb_lower", "bb_width",
                "atr_14"
            ],
            "pred_thresh": 0.5
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    # If model doesn't exist, train it
    if not model_file.exists():
        logger.warning(f"Model file for {symbol} (horizon {horizon}) not found. Training a new model.")
        from PA_ML.crypto_forecast_ml.training.train_model import train_direction_model_with_timerange

        # Train and save model
        train_direction_model_with_timerange(
            symbol=symbol,
            days=30,
            output_path=str(model_file),
            horizon=horizon
        )

        if not model_file.exists():
            logger.error(f"Failed to create model file for {symbol} (horizon {horizon})")
            return False

    try:
        # Load the model
        model_key = f"{symbol}_h{horizon}"
        MODELS[model_key] = xgb.Booster()
        MODELS[model_key].load_model(str(model_file))

        # Store metadata
        METADATA[model_key] = metadata

        logger.info(f"Model loaded: {model_file}")
        return True
    except Exception as e:
        logger.error(f"Error loading model for {symbol} (horizon {horizon}): {str(e)}")
        return False

def predict(symbol: str = "BTCUSDT", use_incomplete_candle: bool = True) -> dict:
    """
    Generate a prediction + trade suggestion for the next 5 minutes.
    This function mimics the behavior of crypto_signals.src.predict.predict().

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        use_incomplete_candle: If True, use the current incomplete candle

    Returns:
        dict: Prediction and trade suggestion
    """
    # Load data
    df = load_crypto_data(symbol, days=1)

    # Add features
    df = add_all_features(df)

    # Extract the last row for prediction
    if use_incomplete_candle:
        feats_last = df.iloc[-1:]
    else:
        feats_last = df.iloc[-2:-1]

    # Select features for prediction
    features = feats_last[[
        "open", "high", "low", "close", "volume", "quote_volume", "nb_trades",
        "sma_5", "sma_10", "ema_5", "ema_10", "rsi_14",
        "macd", "macd_signal", "macd_diff",
        "bb_upper", "bb_lower", "bb_width",
        "atr_14"
    ]]

    if USE_AGGREGATED_MODEL:
        # Use the aggregated model for 5-minute predictions
        model_key = f"{symbol}_5min"

        # Load the model if not already loaded
        if model_key not in MODELS:
            if not load_aggregated_model(symbol):
                return {"error": f"Failed to load aggregated model for {symbol}"}

        # Make prediction
        model = MODELS[model_key]
        dmatrix = xgb.DMatrix(features)
        p_up = model.predict(dmatrix)[0]

        # Get prediction threshold from metadata
        pred_thresh = METADATA[model_key].get("pred_thresh", 0.5)

        # Note for response
        note = "XGBoost prediction using aggregated 5-minute model"
    else:
        # Load models for horizons 1-5 if not already loaded
        models_loaded = True
        for horizon in range(1, 6):
            model_key = f"{symbol}_h{horizon}"
            if model_key not in MODELS:
                if not load_model(symbol, horizon):
                    models_loaded = False
                    logger.error(f"Failed to load model for {symbol} (horizon {horizon})")

        if not models_loaded:
            return {"error": f"Failed to load one or more models for {symbol}"}

        # Make predictions for each horizon
        predictions = []
        for horizon in range(1, 6):
            model_key = f"{symbol}_h{horizon}"
            model = MODELS[model_key]

            dmatrix = xgb.DMatrix(features)
            raw_pred = model.predict(dmatrix)[0]

            # Store prediction
            predictions.append(raw_pred)

        # Aggregate predictions for 5-minute forecast
        # Simple approach: average the predictions
        p_up = sum(predictions) / len(predictions)

        # Get prediction threshold from metadata
        pred_thresh = METADATA[f"{symbol}_h1"].get("pred_thresh", 0.5)

        # Note for response
        note = "XGBoost prediction aggregated from 5 models (t+1 to t+5)"

    # Calculate confidence (distance from 0.5)
    confidence = min(abs(p_up - 0.5) * 2, 1.0)

    # Determine signal
    signal = "LONG" if p_up >= pred_thresh else "SHORT"
    last_price = df["close"].iloc[-1]

    # Calculate stop loss and take profit levels
    # Use ATR if available for dynamic levels
    if "atr_14" in feats_last.columns:
        atr = feats_last["atr_14"].iloc[0]
        sl_multiplier = 1.0
        tp_multiplier = 2.0

        sl_pct = (atr / last_price) * sl_multiplier
        tp_pct = (atr / last_price) * tp_multiplier
    else:
        # Fallback to fixed values
        sl_pct = 0.002  # 0.2%
        tp_pct = 0.004  # 0.4%

    # Calculate entry, SL, and TP levels
    if signal == "LONG":
        entry = last_price
        stop_loss = round(entry * (1 - sl_pct), 4)
        take_profit = round(entry * (1 + tp_pct), 4)
    else:
        entry = last_price
        stop_loss = round(entry * (1 + sl_pct), 4)
        take_profit = round(entry * (1 - tp_pct), 4)

    # Build response
    return {
        "symbol": symbol,
        "timestamp": df["timestamp_utc"].iloc[-1].isoformat(),
        "prob_up": round(float(p_up), 4),
        "signal": signal,
        "confidence": round(float(confidence), 4),
        "using_incomplete_candle": use_incomplete_candle,
        "entry": round(float(entry), 4),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "note": note
    }

def get_historical_predictions(symbol: str, days: int = 7) -> List[dict]:
    """
    Generate predictions for historical data.
    This function mimics the behavior of crypto_signals.src.api.get_historical_predictions().

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        days: Number of days of historical data

    Returns:
        List[dict]: List of predictions
    """
    # Load data
    df = load_crypto_data(symbol, days=days)

    # Add features
    df = add_all_features(df)

    # Generate predictions
    predictions = []

    if USE_AGGREGATED_MODEL:
        # Use the aggregated model for 5-minute predictions
        model_key = f"{symbol}_5min"

        # Load the model if not already loaded
        if model_key not in MODELS:
            if not load_aggregated_model(symbol):
                return [{"error": f"Failed to load aggregated model for {symbol}"}]

        model = MODELS[model_key]
        pred_thresh = METADATA[model_key].get("pred_thresh", 0.5)

        for i in range(len(df)):
            try:
                row = df.iloc[i:i+1]

                # Create DMatrix for prediction
                features = row[[
                    "open", "high", "low", "close", "volume", "quote_volume", "nb_trades",
                    "sma_5", "sma_10", "ema_5", "ema_10", "rsi_14",
                    "macd", "macd_signal", "macd_diff",
                    "bb_upper", "bb_lower", "bb_width",
                    "atr_14"
                ]]

                dmatrix = xgb.DMatrix(features)
                p_up = model.predict(dmatrix)[0]

                # Determine signal
                signal = "LONG" if p_up > 0.65 else "SHORT" if p_up < 0.35 else "FLAT"
                confidence = min(abs(p_up - 0.5) * 2, 1.0)

                predictions.append({
                    "symbol": symbol,
                    "timestamp": df["timestamp_utc"].iloc[i].isoformat(),
                    "prob_up": round(float(p_up), 4),
                    "signal": signal,
                    "confidence": round(float(confidence), 4),
                    "model_type": "aggregated_5min"
                })
            except Exception as e:
                logger.error(f"Error generating prediction for row {i}: {str(e)}")
                # Continue with next row
    else:
        # Load models for horizons 1-5 if not already loaded
        models_loaded = True
        for horizon in range(1, 6):
            model_key = f"{symbol}_h{horizon}"
            if model_key not in MODELS:
                if not load_model(symbol, horizon):
                    models_loaded = False
                    logger.error(f"Failed to load model for {symbol} (horizon {horizon})")

        if not models_loaded:
            return [{"error": f"Failed to load one or more models for {symbol}"}]

        # Generate predictions
        for i in range(len(df)):
            try:
                row = df.iloc[i:i+1]

                # Make predictions for each horizon
                horizon_preds = []
                for horizon in range(1, 6):
                    model_key = f"{symbol}_h{horizon}"
                    model = MODELS[model_key]

                    # Create DMatrix for prediction
                    features = row[[
                        "open", "high", "low", "close", "volume", "quote_volume", "nb_trades",
                        "sma_5", "sma_10", "ema_5", "ema_10", "rsi_14",
                        "macd", "macd_signal", "macd_diff",
                        "bb_upper", "bb_lower", "bb_width",
                        "atr_14"
                    ]]

                    dmatrix = xgb.DMatrix(features)
                    raw_pred = model.predict(dmatrix)[0]
                    horizon_preds.append(raw_pred)

                # Aggregate predictions
                p_up = sum(horizon_preds) / len(horizon_preds)

                # Determine signal
                pred_thresh = METADATA[f"{symbol}_h1"].get("pred_thresh", 0.5)
                signal = "LONG" if p_up > 0.65 else "SHORT" if p_up < 0.35 else "FLAT"
                confidence = min(abs(p_up - 0.5) * 2, 1.0)

                predictions.append({
                    "symbol": symbol,
                    "timestamp": df["timestamp_utc"].iloc[i].isoformat(),
                    "prob_up": round(float(p_up), 4),
                    "signal": signal,
                    "confidence": round(float(confidence), 4),
                    "model_type": "multi_horizon"
                })
            except Exception as e:
                logger.error(f"Error generating prediction for row {i}: {str(e)}")
                # Continue with next row

    return predictions

def get_available_models() -> List[str]:
    """
    Get list of available models.
    This function mimics the behavior of crypto_signals.src.api.get_available_models().

    Returns:
        List[str]: List of available models
    """
    model_files = list(MODEL_DIR.glob("xgb_direction_h1_*.json"))
    return [f.name.replace("xgb_direction_h1_", "").replace(".json", "") for f in model_files if f.exists()]

def get_model_metadata(symbol: str) -> dict:
    """
    Get metadata for a specific model.
    This function mimics the behavior of crypto_signals.src.api.get_model_metadata().

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)

    Returns:
        dict: Model metadata
    """
    metadata_file = MODEL_DIR / f"metadata_h1_{symbol}.json"
    if not metadata_file.exists():
        return {"error": f"No metadata found for {symbol}"}

    try:
        with open(metadata_file, "r") as f:
            file_content = f.read().strip()
            if file_content:  # Check if file is not empty
                return json.loads(file_content)
            else:
                logger.warning(f"Metadata file {metadata_file} is empty.")
                return {"error": f"Metadata file for {symbol} is empty"}
    except json.JSONDecodeError as e:
        logger.warning(f"Error loading metadata file {metadata_file}: {str(e)}")
        return {"error": f"Invalid JSON in metadata file for {symbol}: {str(e)}"}

def get_model_info(symbol: str) -> dict:
    """
    Get detailed information about a specific model.
    This function mimics the behavior of crypto_signals.src.api.get_model_info().

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)

    Returns:
        dict: Model information
    """
    # Check if model file exists
    model_file = MODEL_DIR / f"xgb_direction_h1_{symbol}.json"
    if not model_file.exists():
        return {"error": f"Model file for {symbol} not found"}

    # Load model if not already loaded
    model_key = f"{symbol}_h1"
    if model_key not in MODELS:
        if not load_model(symbol, 1):
            return {"error": f"Failed to load model for {symbol}"}

    # Get metadata
    metadata = get_model_metadata(symbol)

    # Get feature importance
    feature_importance = {}
    try:
        model = MODELS[model_key]
        importance = model.get_score(importance_type='gain')
        feature_importance = importance
    except Exception as e:
        logger.warning(f"Could not get feature importance for {symbol}: {str(e)}")

    return {
        "symbol": symbol,
        "metadata": metadata,
        "feature_importance": feature_importance,
        "features": metadata.get("features", [])
    }

def load_last_candle(symbol: str = "BTCUSDT") -> dict:
    """
    Retrieve only the last (most recent) candle for a given symbol.
    This function mimics the behavior of crypto_signals.src.data_loader.load_last_candle().

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)

    Returns:
        dict: Last candle data
    """
    try:
        df = load_crypto_data(symbol, days=1)
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return {}

        # Convert to dictionary and format timestamp
        candle = df.iloc[-1].to_dict()
        candle["timestamp_utc"] = candle["timestamp_utc"].isoformat()

        logger.info(f"{symbol} – last candle loaded: {candle['timestamp_utc']}")
        return candle
    except Exception as e:
        logger.error(f"Error loading last candle for {symbol}: {str(e)}")
        return {}

def get_historical_data(symbol: str, days: int = 7, interval: str = "1m", raw: bool = False) -> Union[List[dict], dict]:
    """
    Return historical candle data.
    This function mimics the behavior of crypto_signals.src.api.get_historical_data().

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        days: Number of days of historical data
        interval: Data interval (1m, 1h)
        raw: If True, return plain array instead of wrapped object

    Returns:
        Union[List[dict], dict]: Historical data
    """
    try:
        df = load_crypto_data(symbol, days=days)

        if interval == "1h":
            # Resample to hourly data
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
                "nb_trades": "sum"
            }
            df = (df
                  .set_index("timestamp_utc")
                  .resample("1H")
                  .agg(agg)
                  .dropna()
                  .reset_index())

        candles = df.to_dict(orient="records")
        for item in candles:
            item["timestamp_utc"] = item["timestamp_utc"].isoformat()

        if raw:
            return candles

        return {
            "symbol": symbol,
            "interval": interval,
            "days": days,
            "count": len(candles),
            "data": candles
        }
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return {"error": f"Failed to load data: {str(e)}"}
