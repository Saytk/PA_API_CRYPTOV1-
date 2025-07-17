# crypto_forecast_ml/refresh_model.py

from crypto_forecast_ml.data_loader import load_crypto_data
from crypto_forecast_ml.features.technical_indicators import add_technical_indicators
from crypto_forecast_ml.features.target_builder import build_targets
from crypto_forecast_ml.training.train_model import train_direction_model, train_direction_model_with_timerange
from crypto_forecast_ml.predictor.predict import predict_direction

import pandas as pd
import xgboost as xgb
import os
import logging
from math import log

# ⚙️ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_log_loss(y_true, y_pred, eps=1e-15):
    """
    Calcule le log loss sans sklearn.
    """
    y_pred = [max(min(p, 1 - eps), eps) for p in y_pred]
    loss = -sum(y * log(p) + (1 - y) * log(1 - p) for y, p in zip(y_true, y_pred)) / len(y_true)
    return loss

def evaluate_model(df: pd.DataFrame, model_path: str) -> float:
    df = df.dropna().copy()
    X = df.drop(columns=["timestamp_utc", "next_close", "return_next", "direction"])
    y_true = df["direction"].tolist()
    dmatrix = xgb.DMatrix(X)

    model = xgb.Booster()
    model.load_model(str(model_path))

    y_pred = model.predict(dmatrix).tolist()
    return compute_log_loss(y_true, y_pred)

def refresh_model(symbol: str = "BTCUSDT", hours: int = None, days: int = 3, all_data: bool = False):
    """
    Rafraîchit le modèle en entraînant sur les données récentes.

    Args:
        symbol (str): Symbole de la paire de trading (ex: "BTCUSDT")
        hours (int, optional): Nombre d'heures de données à utiliser.
                              Si spécifié, remplace le paramètre days.
        days (int): Nombre de jours de données à utiliser si hours n'est pas spécifié
        all_data (bool): Si True, utilise toutes les données disponibles sans contrainte de temps
    """
    logger.info("🔁 Refresh model process started")
    logger.info(f"Symbol: {symbol}, Hours: {hours}, Days: {days}, All data: {all_data}")

    temp_model_path = "models/xgb_direction_temp.json"
    final_model_path = "models/xgb_direction.json"

    # Utilisation de la nouvelle fonction avec plage horaire
    df = train_direction_model_with_timerange(
        symbol=symbol,
        hours=hours,
        days=days,
        all_data=all_data,
        output_path=temp_model_path
    )

    try:
        old_score = evaluate_model(df, final_model_path)
    except Exception:
        old_score = float("inf")
        logger.warning("⚠️ Aucun ancien modèle valide trouvé, on utilisera le nouveau directement.")

    new_score = evaluate_model(df, temp_model_path)
    logger.info(f"📊 Old logloss: {old_score:.5f} — New logloss: {new_score:.5f}")

    # Obtenir les chemins absolus pour le logging
    if os.path.isabs(final_model_path):
        abs_final_path = final_model_path
    else:
        # Si le chemin est relatif, le convertir en absolu
        current_dir = os.path.dirname(os.path.abspath(__file__))
        abs_final_path = os.path.join(current_dir, final_model_path)

    if new_score < old_score:
        os.replace(temp_model_path, final_model_path)
        logger.info(f"✅ Nouveau modèle adopté ✅ Sauvegardé dans: {abs_final_path}")
    else:
        os.remove(temp_model_path)
        logger.info("❌ Nouveau modèle rejeté — moins performant")

if __name__ == "__main__":
    import argparse

    # Parsing des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Script de rafraîchissement du modèle de prédiction crypto")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbole de la paire (ex: BTCUSDT)")
    parser.add_argument("--hours", type=int, help="Nombre d'heures de données à utiliser pour l'entraînement")
    parser.add_argument("--days", type=int, default=3, help="Nombre de jours de données si hours n'est pas spécifié")
    parser.add_argument("--all-data", action="store_true", help="Utiliser toutes les données disponibles sans contrainte de temps")

    args = parser.parse_args()

    # Appel de la fonction avec les paramètres spécifiés
    refresh_model(
        symbol=args.symbol,
        hours=args.hours,
        days=args.days,
        all_data=args.all_data
    )
