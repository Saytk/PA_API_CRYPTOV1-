import pandas as pd
import xgboost as xgb
import os
import logging
from pathlib import Path

from PA_ML.crypto_forecast_ml.training.train_model import train_direction_model
from PA_ML.crypto_forecast_ml.features.feature_engineering import add_all_features

logger = logging.getLogger(__name__)

def predict_direction(df: pd.DataFrame) -> pd.DataFrame:
    # Use absolute path for model file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "xgb_direction.json")
    logger.info(f"Using model path: {model_path}")

    if not os.path.exists(model_path):
        logger.warning(f"⚠️ Model not found at {model_path}, training a new one...")
        df = add_all_features(df)
        train_direction_model(df, model_path)
        logger.info("✅ New model trained and saved.")

    df = add_all_features(df)

    features = df[[  # les features doivent être dans le même ordre qu’à l’entraînement
        "open", "high", "low", "close", "volume", "quote_volume", "nb_trades",
        "sma_5", "sma_10", "ema_5", "ema_10", "rsi_14",
        "macd", "macd_signal", "macd_diff",
        "bb_upper", "bb_lower", "bb_width",
        "atr_14"
    ]].copy()

    model = xgb.Booster()
    model.load_model(str(model_path))
    logger.info("✅ XGBoost model loaded successfully.")

    dmatrix = xgb.DMatrix(features)
    preds = model.predict(dmatrix)

    df["prediction"] = preds
    return df[["timestamp_utc", "close", "prediction"]]
