# features/target_builder.py

import pandas as pd
import numpy as np

def build_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Ajoute les colonnes 'next_close', 'return_next', 'direction' comme targets.
    Supporte différents horizons de prédiction (1 à 5 minutes).

    Args:
        df (pd.DataFrame): OHLCV + indicateurs
        horizon (int): Combien de périodes dans le futur on prédit (1-5)

    Returns:
        pd.DataFrame: DataFrame avec colonnes cibles
    """
    df = df.copy()

    # Valeur future de close
    df[f"next_close_h{horizon}"] = df["close"].shift(-horizon)

    # Rendement simple
    df[f"return_next_h{horizon}"] = (df[f"next_close_h{horizon}"] - df["close"]) / df["close"]

    # Direction (classification binaire)
    df[f"direction_h{horizon}"] = (df[f"return_next_h{horizon}"] > 0).astype(int)

    # Pour la compatibilité avec le code existant, on garde aussi les noms originaux
    if horizon == 1:
        df["next_close"] = df[f"next_close_h{horizon}"]
        df["return_next"] = df[f"return_next_h{horizon}"]
        df["direction"] = df[f"direction_h{horizon}"]

    # Supprime les lignes avec NaN dans les colonnes cibles
    target_cols = [f"next_close_h{horizon}", f"return_next_h{horizon}", f"direction_h{horizon}"]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    return df
