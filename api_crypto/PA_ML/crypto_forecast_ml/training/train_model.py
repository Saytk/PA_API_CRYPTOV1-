# crypto_forecast_ml/training/train_model.py

import pandas as pd
import xgboost as xgb
import os
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta

# ⚙️ Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_direction_model(df: pd.DataFrame, output_path: str = "models/xgb_direction.json", horizon: int = 1):
    """
    Entraîne un modèle XGBoost pour prédire la direction (hausse/baisse) sans sklearn.

    Args:
        df (pd.DataFrame): Données avec indicateurs et colonnes cibles
        output_path (str): Chemin pour sauvegarder le modèle
        horizon (int): Horizon de prédiction (1-5)
    """
    logger.info(f"Colonnes à l'entrée de train_direction_model: {df.columns.tolist()}")
    logger.info(f"Nombre de lignes à l'entrée de train_direction_model: {len(df)}")

    df = df.dropna().copy()
    logger.info(f"Colonnes après dropna: {df.columns.tolist()}")
    logger.info(f"Nombre de lignes après dropna: {len(df)}")

    # Sélection des features
    try:
        # Vérifier si toutes les colonnes nécessaires sont présentes
        target_col = f"direction_h{horizon}"
        if horizon == 1:
            # Pour horizon 1, vérifier aussi les colonnes originales
            if "direction" not in df.columns:
                raise KeyError(f"Colonne manquante: direction")
            target_col = "direction"  # Utiliser la colonne originale pour horizon 1
        elif target_col not in df.columns:
            raise KeyError(f"Colonne manquante: {target_col}")

        # Exclure toutes les colonnes cibles et de timestamp
        cols_to_drop = ["timestamp_utc"]
        for h in range(1, 6):
            # Exclure les colonnes pour tous les horizons
            for prefix in ["next_close_h", "return_next_h", "direction_h"]:
                col = f"{prefix}{h}"
                if col in df.columns:
                    cols_to_drop.append(col)

        # Exclure aussi les colonnes originales
        for col in ["next_close", "return_next", "direction"]:
            if col in df.columns:
                cols_to_drop.append(col)

        X = df.drop(columns=cols_to_drop)
        y = df[target_col]
    except Exception as e:
        logger.error(f"Erreur lors de la sélection des features: {str(e)}")
        raise

    # Encodage dans un DMatrix (XGBoost natif)
    dtrain = xgb.DMatrix(X, label=y)

    # Paramètres XGBoost
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 1
    }

    # Entraînement
    model = xgb.train(params, dtrain, num_boost_round=100)

    # Sauvegarde du modèle
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # Obtenir le chemin absolu pour le logging
    if os.path.isabs(output_path):
        abs_path = output_path
    else:
        # Si le chemin est relatif, le convertir en absolu
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_dir))
        abs_path = os.path.join(base_dir, output_path)

    model.save_model(str(output_path))
    logger.info(f"✅ Modèle entraîné et sauvegardé dans : {output_path}")
    logger.info(f"✅ Chemin absolu du modèle : {abs_path}")

def train_direction_model_with_timerange(symbol: str = "BTCUSDT", hours: int = None, days: int = 7, all_data: bool = False, output_path: str = "models/xgb_direction.json", horizon: int = 1):
    """
    Charge les données pour une plage horaire spécifique, puis entraîne un modèle XGBoost.

    Args:
        symbol (str): Symbole de la paire de trading (ex: "BTCUSDT")
        hours (int, optional): Nombre d'heures de données à utiliser pour l'entraînement.
                              Si spécifié, remplace le paramètre days.
        days (int): Nombre de jours de données à utiliser si hours n'est pas spécifié
        all_data (bool): Si True, utilise toutes les données disponibles sans contrainte de temps
        output_path (str): Chemin pour sauvegarder le modèle
        horizon (int): Horizon de prédiction (1-5)
    """
    from PA_ML.crypto_forecast_ml.data_loader import load_crypto_data, load_crypto_data_custom_range, load_crypto_data_all
    from PA_ML.crypto_forecast_ml.features.feature_engineering import add_all_features
    from PA_ML.crypto_forecast_ml.features.target_builder import build_targets

    logger.info(f"🚀 Démarrage de l'entraînement avec plage horaire personnalisée")
    logger.info(f"Symbol: {symbol}, Hours: {hours}, Days: {days}, All data: {all_data}, Horizon: {horizon}")

    if all_data:
        # Utilisation de toutes les données disponibles
        logger.info(f"📅 Utilisation de TOUTES les données disponibles")
        df = load_crypto_data_all(symbol)
    elif hours is not None:
        # Calcul des dates pour la plage horaire spécifiée
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)

        # Format des dates pour BigQuery
        start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"📅 Plage horaire: {start_date_str} à {end_date_str} (UTC)")

        # Chargement des données avec plage personnalisée
        df = load_crypto_data_custom_range(symbol, start_date_str, end_date_str)
    else:
        # Utilisation de la méthode standard avec nombre de jours
        logger.info(f"📅 Utilisation des {days} derniers jours de données")
        df = load_crypto_data(symbol, days=days)

    # Feature engineering
    logger.info(f"🔧 Application du feature engineering...")
    df = add_all_features(df)

    # Construction des cibles
    logger.info(f"🎯 Construction des variables cibles...")
    df = build_targets(df, horizon=horizon)

    # Vérification des données
    logger.info(f"📊 Nombre de lignes pour l'entraînement: {len(df)}")

    # Entraînement du modèle
    logger.info(f"⚙️ Entraînement du modèle...")
    train_direction_model(df, output_path, horizon=horizon)

    return df

def train_aggregated_model(symbol: str = "BTCUSDT", days: int = 30, output_path: str = "models/xgb_direction_5min.json"):
    """
    Entraîne un modèle unique qui prédit la direction du prix sur 5 minutes.
    Ce modèle est une alternative plus simple aux 5 modèles séparés.

    Args:
        symbol (str): Symbole de la paire de trading (ex: "BTCUSDT")
        days (int): Nombre de jours de données à utiliser
        output_path (str): Chemin pour sauvegarder le modèle
    """
    from PA_ML.crypto_forecast_ml.data_loader import load_crypto_data
    from PA_ML.crypto_forecast_ml.features.feature_engineering import add_all_features
    import pandas as pd

    logger.info(f"🚀 Démarrage de l'entraînement du modèle agrégé 5 minutes")
    logger.info(f"Symbol: {symbol}, Days: {days}")

    # Chargement des données
    df = load_crypto_data(symbol, days=days)

    # Feature engineering
    df = add_all_features(df)

    # Création d'une cible spécifique pour 5 minutes
    # On regarde si le prix est plus haut 5 minutes plus tard
    df["next_close_5min"] = df["close"].shift(-5)
    df["return_5min"] = (df["next_close_5min"] - df["close"]) / df["close"]
    df["direction_5min"] = (df["return_5min"] > 0).astype(int)

    # Suppression des lignes avec NaN
    df = df.dropna(subset=["direction_5min"]).reset_index(drop=True)

    # Sélection des features (exclure toutes les colonnes cibles)
    cols_to_drop = ["timestamp_utc", "next_close_5min", "return_5min", "direction_5min"]

    # Exclure aussi les colonnes pour tous les horizons et les colonnes originales
    for h in range(1, 6):
        for prefix in ["next_close_h", "return_next_h", "direction_h"]:
            col = f"{prefix}{h}"
            if col in df.columns:
                cols_to_drop.append(col)

    for col in ["next_close", "return_next", "direction"]:
        if col in df.columns:
            cols_to_drop.append(col)

    X = df.drop(columns=cols_to_drop)
    y = df["direction_5min"]

    # Encodage dans un DMatrix
    dtrain = xgb.DMatrix(X, label=y)

    # Paramètres XGBoost
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 1
    }

    # Entraînement
    model = xgb.train(params, dtrain, num_boost_round=100)

    # Sauvegarde du modèle
    os.makedirs(Path(output_path).parent, exist_ok=True)
    model.save_model(str(output_path))

    # Sauvegarde des métadonnées
    metadata = {
        "symbol": symbol,
        "created_at": datetime.now().isoformat(),
        "model_type": "xgboost_5min",
        "features": X.columns.tolist(),
        "pred_thresh": 0.5,
        "description": "Modèle agrégé pour prédiction sur 5 minutes"
    }

    metadata_path = str(output_path).replace(".json", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Modèle agrégé 5 minutes entraîné et sauvegardé dans : {output_path}")
    logger.info(f"✅ Métadonnées sauvegardées dans : {metadata_path}")

    return df
