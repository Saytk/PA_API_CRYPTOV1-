# Unified Crypto API

This API integrates functionality from both PA_ML and crypto_signals modules, providing a comprehensive interface for cryptocurrency analysis, prediction, pattern detection, and portfolio management.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Running the API](#running-the-api)
4. [Authentication](#authentication)
5. [API Endpoints](#api-endpoints)
   - [Prediction Endpoints](#prediction-endpoints)
   - [Pattern Detection Endpoints](#pattern-detection-endpoints)
   - [Portfolio Management Endpoints](#portfolio-management-endpoints)
   - [Data Endpoints](#data-endpoints)
6. [Changes Made to the Project](#changes-made-to-the-project)

## Overview

The Unified Crypto API combines the functionality of two separate modules:

1. **crypto_signals**: Provides ML-based predictions and model management
2. **PA_ML**: Provides pattern detection in candlestick charts

Additionally, it adds new portfolio management functionality to allow users to track their cryptocurrency holdings.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the API

To start the API server, run:

```bash
python -m crypto_api.run_api
```

The API will be available at `http://localhost:8000`.

## Authentication

The API uses OAuth2 for authentication. To authenticate:

1. Make a POST request to `/token` with your username and password
2. Use the returned token in the Authorization header for subsequent requests

Example:

```bash
# Get token
curl -X POST "http://localhost:8000/token" -d "username=admin&password=secret"

# Use token
curl -X GET "http://localhost:8000/portfolio" -H "Authorization: Bearer YOUR_TOKEN"
```

## API Endpoints

### Prediction Endpoints

These endpoints provide ML-based predictions for cryptocurrency price movements.

#### GET /prediction/latest

Returns the probability that the next minute candle will close higher, along with a discrete LONG / FLAT / SHORT signal.

**Parameters:**
- `symbol` (string, default: "BTCUSDT"): Trading pair symbol
- `use_incomplete_candle` (boolean, default: true): Whether to use the incomplete current candle

**Example:**
```bash
curl -X GET "http://localhost:8000/prediction/latest?symbol=BTCUSDT&use_incomplete_candle=true"
```

#### GET /prediction/historical/{symbol}

Returns historical predictions for a given symbol.

**Parameters:**
- `symbol` (string): Trading pair symbol
- `days` (integer, default: 7): Number of days of historical data

**Example:**
```bash
curl -X GET "http://localhost:8000/prediction/historical/BTCUSDT?days=7"
```

#### GET /prediction/models

Returns the list of available prediction models.

**Example:**
```bash
curl -X GET "http://localhost:8000/prediction/models"
```

#### GET /prediction/models/{symbol}

Returns detailed information about a specific model.

**Parameters:**
- `symbol` (string): Trading pair symbol

**Example:**
```bash
curl -X GET "http://localhost:8000/prediction/models/BTCUSDT"
```

#### POST /prediction/backtest

Runs a walk-forward backtest and returns the results. Requires authentication.

**Parameters:**
- `symbol` (string, default: "BTCUSDT"): Trading pair symbol
- `n_folds` (integer, default: 5): Number of folds for walk-forward validation

**Example:**
```bash
curl -X POST "http://localhost:8000/prediction/backtest?symbol=BTCUSDT&n_folds=5" -H "Authorization: Bearer YOUR_TOKEN"
```

#### POST /prediction/train

Triggers training of a new model (background task). Requires authentication.

**Parameters:**
- `symbol` (string, default: "BTCUSDT"): Trading pair symbol

**Example:**
```bash
curl -X POST "http://localhost:8000/prediction/train?symbol=BTCUSDT" -H "Authorization: Bearer YOUR_TOKEN"
```

### Pattern Detection Endpoints

These endpoints provide pattern detection in candlestick charts.

#### GET /pattern/predict-latest

Returns the latest pattern prediction for a given symbol.

**Parameters:**
- `symbol` (string, default: "BTCUSDT"): Crypto symbol

**Example:**
```bash
curl -X GET "http://localhost:8000/pattern/predict-latest?symbol=BTCUSDT"
```

#### GET /pattern/load-data

Loads historical data for pattern analysis.

**Parameters:**
- `symbol` (string): Crypto symbol
- `start_date` (string): Start date in format YYYY-MM-DDTHH:MM
- `end_date` (string): End date in format YYYY-MM-DDTHH:MM

**Example:**
```bash
curl -X GET "http://localhost:8000/pattern/load-data?symbol=BTCUSDT&start_date=2023-01-01T00:00&end_date=2023-01-07T00:00"
```

#### GET /pattern/load-data-patterns

Loads historical data and detects patterns within it.

**Parameters:**
- `symbol` (string): Crypto symbol
- `start_date` (string): Start date in format YYYY-MM-DDTHH:MM
- `end_date` (string): End date in format YYYY-MM-DDTHH:MM

**Example:**
```bash
curl -X GET "http://localhost:8000/pattern/load-data-patterns?symbol=BTCUSDT&start_date=2023-01-01T00:00&end_date=2023-01-07T00:00"
```

### Portfolio Management Endpoints

These endpoints provide portfolio management functionality. All require authentication.

#### GET /portfolio

Get the current user's portfolio.

**Example:**
```bash
curl -X GET "http://localhost:8000/portfolio" -H "Authorization: Bearer YOUR_TOKEN"
```

#### POST /portfolio/transaction

Add a buy/sell transaction to the portfolio.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "amount": 0.1,
  "price": 50000,
  "transaction_type": "buy"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/portfolio/transaction" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_TOKEN" -d '{"symbol":"BTCUSDT","amount":0.1,"price":50000,"transaction_type":"buy"}'
```

#### DELETE /portfolio

Reset the current user's portfolio.

**Example:**
```bash
curl -X DELETE "http://localhost:8000/portfolio" -H "Authorization: Bearer YOUR_TOKEN"
```

### Data Endpoints

#### GET /data/{symbol}

Returns historical data for a given symbol.

**Parameters:**
- `symbol` (string): Trading pair symbol
- `days` (integer, default: 7): Number of days of historical data
- `interval` (string, default: "1m"): Data interval (1m, 1h)

**Example:**
```bash
curl -X GET "http://localhost:8000/data/BTCUSDT?days=7&interval=1m"
```

## Changes Made to the Project

The following changes were made to integrate the PA_ML and crypto_signals modules:

1. Created a new `crypto_api` package to house the unified API
2. Implemented a FastAPI application that integrates functionality from both modules
3. Created separate routers for different types of functionality:
   - Prediction router for ML-based predictions from crypto_signals
   - Pattern router for pattern detection from PA_ML
   - Portfolio router for new portfolio management functionality
4. Reused authentication from crypto_signals
5. Added comprehensive documentation for the API

The integration was done in a way that minimizes changes to the existing codebase. Instead of modifying the original modules, we import and use their functionality directly in the unified API. This approach ensures that the original modules continue to work as before, while also providing a unified interface for all functionality.