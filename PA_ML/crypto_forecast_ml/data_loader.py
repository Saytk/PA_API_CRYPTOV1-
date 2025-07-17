# crypto_forecast_ml/data_loader.py

import os
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from functools import lru_cache
from typing import Dict, Tuple, Optional, Any
import json

from google.oauth2 import service_account

# ⚙️ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for storing query results
# Structure: {cache_key: (timestamp, dataframe)}
_DATA_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
# Cache expiration time in seconds (5 minutes)
_CACHE_EXPIRY = 300

def load_crypto_data(symbol: str = "BTCUSDT", days: int = 7, max_rows: int = 100_000) -> pd.DataFrame:
    logger.info("🟡 load_crypto_data() CALLED")

    # Create a cache key based on function parameters
    cache_key = f"load_crypto_data_{symbol}_{days}_{max_rows}"

    # Check if data is in cache and not expired
    current_time = time.time()
    if cache_key in _DATA_CACHE:
        cache_time, cached_df = _DATA_CACHE[cache_key]
        if current_time - cache_time < _CACHE_EXPIRY:
            logger.info(f"✅ Using cached data for {symbol} (cached {int(current_time - cache_time)} seconds ago)")
            return cached_df
        else:
            logger.info(f"🔄 Cache expired for {symbol}, fetching fresh data")

    # 🔧 Recherche le chemin absolu vers le dossier crypto_forecast_ml
    gcp_credentials_json = os.environ["GCP_CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_info(json.loads(gcp_credentials_json))
    bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)






    # Fenêtre temporelle
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
    query = f"""
        SELECT timestamp_utc, open, high, low, close, volume, quote_volume, nb_trades
        FROM `feisty-coder-461708-m9.data_bronze.RAW_CRYPTO_KLINES_1MIN`
        WHERE symbol = '{symbol}'
          AND timestamp_utc >= TIMESTAMP('{start_date}')
        ORDER BY timestamp_utc ASC
        LIMIT {max_rows}
    """

    # Log pour le debugging
    logger.info(f"Query: {query}")

    logger.info(f"📥 Launching BigQuery query for {symbol}...")
    df = bq_client.query(query).to_dataframe()
    logger.info(f"📊 Loaded {len(df)} rows.")
    logger.info(f"Colonnes dans le DataFrame: {df.columns.tolist()}")

    # Store in cache
    _DATA_CACHE[cache_key] = (current_time, df)

    return df


def load_crypto_data_custom_range(symbol: str, start_date: str, end_date: str, max_rows: int = 100_000) -> pd.DataFrame:
    logger.info(f"🟡 load_crypto_data_custom_range() CALLED with symbol={symbol}, start={start_date}, end={end_date}")

    # Create a cache key based on function parameters
    cache_key = f"load_crypto_data_custom_range_{symbol}_{start_date}_{end_date}_{max_rows}"

    # Check if data is in cache and not expired
    current_time = time.time()
    if cache_key in _DATA_CACHE:
        cache_time, cached_df = _DATA_CACHE[cache_key]
        if current_time - cache_time < _CACHE_EXPIRY:
            logger.info(f"✅ Using cached data for {symbol} (cached {int(current_time - cache_time)} seconds ago)")
            return cached_df
        else:
            logger.info(f"🔄 Cache expired for {symbol}, fetching fresh data")



    gcp_credentials_json = os.environ["GCP_CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_info(json.loads(gcp_credentials_json))
    bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    # BigQuery client


    query = f"""
        SELECT timestamp_utc, open, high, low, close, volume, quote_volume, nb_trades
        FROM `feisty-coder-461708-m9.data_bronze.RAW_CRYPTO_KLINES_1MIN`
        WHERE symbol = '{symbol}'
          AND timestamp_utc >= TIMESTAMP('{start_date}')
          AND timestamp_utc <= TIMESTAMP('{end_date}')
        ORDER BY timestamp_utc ASC
        LIMIT {max_rows}
    """

    logger.info(f"📥 Running query for range: {start_date} to {end_date}")
    logger.info(f"📥 Running query : {query}")
    df = bq_client.query(query).to_dataframe()
    logger.info(f"📊 Loaded {len(df)} rows.")

    # Store in cache
    _DATA_CACHE[cache_key] = (current_time, df)

    return df


def clear_cache(symbol: Optional[str] = None) -> None:
    """
    Clear the data cache for a specific symbol or all symbols.

    Args:
        symbol (Optional[str]): Symbol to clear cache for. If None, clears entire cache.
    """
    global _DATA_CACHE

    if symbol is None:
        # Clear entire cache
        _DATA_CACHE = {}
        logger.info("🧹 Cleared entire data cache")
    else:
        # Clear cache for specific symbol
        keys_to_remove = [k for k in _DATA_CACHE.keys() if symbol in k]
        for key in keys_to_remove:
            del _DATA_CACHE[key]
        logger.info(f"🧹 Cleared cache for symbol: {symbol} ({len(keys_to_remove)} entries)")


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current cache state.

    Returns:
        Dict[str, Any]: Cache statistics
    """
    symbols = set()
    for key in _DATA_CACHE.keys():
        for part in key.split('_'):
            if part.endswith('USDT'):
                symbols.add(part)

    return {
        "cache_entries": len(_DATA_CACHE),
        "symbols_cached": list(symbols),
        "cache_size_mb": sum(df.memory_usage(deep=True).sum() / (1024 * 1024) 
                            for _, df in [v for v in _DATA_CACHE.values()]),
        "cache_expiry_seconds": _CACHE_EXPIRY
    }


def load_crypto_data_all(symbol: str, max_rows: int = 1_000_000) -> pd.DataFrame:
    """
    Charge toutes les données disponibles pour un symbole donné, sans contrainte de temps.

    Args:
        symbol (str): Symbole de la paire de trading (ex: "BTCUSDT")
        max_rows (int): Nombre maximum de lignes à charger

    Returns:
        pd.DataFrame: DataFrame avec les données OHLCV
    """
    logger.info(f"🟡 load_crypto_data_all() CALLED with symbol={symbol}")

    # Create a cache key based on function parameters
    cache_key = f"load_crypto_data_all_{symbol}_{max_rows}"

    # Check if data is in cache and not expired
    current_time = time.time()
    if cache_key in _DATA_CACHE:
        cache_time, cached_df = _DATA_CACHE[cache_key]
        if current_time - cache_time < _CACHE_EXPIRY:
            logger.info(f"✅ Using cached data for {symbol} (cached {int(current_time - cache_time)} seconds ago)")
            return cached_df
        else:
            logger.info(f"🔄 Cache expired for {symbol}, fetching fresh data")

    gcp_credentials_json = os.environ["GCP_CREDENTIALS"]
    credentials = service_account.Credentials.from_service_account_info(json.loads(gcp_credentials_json))
    bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)


    query = f"""
        SELECT timestamp_utc, open, high, low, close, volume, quote_volume, nb_trades
        FROM `feisty-coder-461708-m9.data_bronze.RAW_CRYPTO_KLINES_1MIN`
        WHERE symbol = '{symbol}'
        ORDER BY timestamp_utc ASC
        LIMIT {max_rows}
    """

    logger.info(f"📥 Running query for ALL available data")
    logger.info(f"📥 Running query : {query}")
    df = bq_client.query(query).to_dataframe()
    logger.info(f"📊 Loaded {len(df)} rows.")

    # Store in cache
    _DATA_CACHE[cache_key] = (current_time, df)

    return df
