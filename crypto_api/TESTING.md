# Testing the Unified Crypto API

This document outlines the testing process for the Unified Crypto API to ensure that all features work as expected and that there are no regressions in existing functionality.

## Prerequisites

Before testing, ensure that all dependencies are installed:

```bash
pip install -r crypto_api/requirements.txt
```

## Starting the API

Start the API server:

```bash
python -m crypto_api.run_api
```

The API will be available at `http://localhost:8000`.

## Testing Authentication

1. Test getting a token:
   ```bash
   curl -X POST "http://localhost:8000/token" -d "username=admin&password=secret"
   ```
   Expected result: A JSON response containing an access token.

2. Test accessing a protected endpoint without authentication:
   ```bash
   curl -X GET "http://localhost:8000/portfolio"
   ```
   Expected result: A 401 Unauthorized error.

3. Test accessing a protected endpoint with authentication:
   ```bash
   curl -X GET "http://localhost:8000/portfolio" -H "Authorization: Bearer YOUR_TOKEN"
   ```
   Expected result: A JSON response containing the user's portfolio.

## Testing Prediction Endpoints

1. Test getting the latest prediction:
   ```bash
   curl -X GET "http://localhost:8000/prediction/latest?symbol=BTCUSDT"
   ```
   Expected result: A JSON response containing the latest prediction for BTCUSDT.

2. Test getting historical predictions:
   ```bash
   curl -X GET "http://localhost:8000/prediction/historical/BTCUSDT?days=1"
   ```
   Expected result: A JSON response containing historical predictions for BTCUSDT.

3. Test getting available models:
   ```bash
   curl -X GET "http://localhost:8000/prediction/models"
   ```
   Expected result: A JSON response containing a list of available models.

4. Test getting model information:
   ```bash
   curl -X GET "http://localhost:8000/prediction/models/BTCUSDT"
   ```
   Expected result: A JSON response containing information about the BTCUSDT model.

5. Test running a backtest (requires authentication):
   ```bash
   curl -X POST "http://localhost:8000/prediction/backtest?symbol=BTCUSDT&n_folds=2" -H "Authorization: Bearer YOUR_TOKEN"
   ```
   Expected result: A JSON response containing backtest results.

## Testing Pattern Detection Endpoints

1. Test getting the latest pattern prediction:
   ```bash
   curl -X GET "http://localhost:8000/pattern/predict-latest?symbol=BTCUSDT"
   ```
   Expected result: A JSON response containing the latest pattern prediction for BTCUSDT.

2. Test loading data for pattern analysis:
   ```bash
   curl -X GET "http://localhost:8000/pattern/load-data?symbol=BTCUSDT&start_date=2023-01-01T00:00&end_date=2023-01-02T00:00"
   ```
   Expected result: A JSON response containing historical data for BTCUSDT.

3. Test loading data with pattern detection:
   ```bash
   curl -X GET "http://localhost:8000/pattern/load-data-patterns?symbol=BTCUSDT&start_date=2023-01-01T00:00&end_date=2023-01-02T00:00"
   ```
   Expected result: A JSON response containing historical data with detected patterns for BTCUSDT.

## Testing Portfolio Management Endpoints

1. Test getting the portfolio (requires authentication):
   ```bash
   curl -X GET "http://localhost:8000/portfolio" -H "Authorization: Bearer YOUR_TOKEN"
   ```
   Expected result: A JSON response containing the user's portfolio.

2. Test adding a transaction (requires authentication):
   ```bash
   curl -X POST "http://localhost:8000/portfolio/transaction" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_TOKEN" -d '{"symbol":"BTCUSDT","amount":0.1,"price":50000,"transaction_type":"buy"}'
   ```
   Expected result: A JSON response containing the updated portfolio.

3. Test resetting the portfolio (requires authentication):
   ```bash
   curl -X DELETE "http://localhost:8000/portfolio" -H "Authorization: Bearer YOUR_TOKEN"
   ```
   Expected result: A JSON response containing an empty portfolio.

## Testing Data Endpoints

1. Test getting historical data:
   ```bash
   curl -X GET "http://localhost:8000/data/BTCUSDT?days=1&interval=1m"
   ```
   Expected result: A JSON response containing historical data for BTCUSDT.

## Regression Testing

To ensure that there are no regressions in existing functionality, compare the responses from the unified API with the responses from the original APIs:

1. Compare the response from `/prediction/latest` with the response from the original crypto_signals API.
2. Compare the response from `/pattern/predict-latest` with the response from the original PA_ML API.
3. Compare the response from `/data/BTCUSDT` with the response from the original crypto_signals API.

## Error Handling Testing

Test error handling by providing invalid inputs:

1. Test with an invalid symbol:
   ```bash
   curl -X GET "http://localhost:8000/prediction/latest?symbol=INVALID"
   ```
   Expected result: An appropriate error message.

2. Test with invalid date formats:
   ```bash
   curl -X GET "http://localhost:8000/pattern/load-data?symbol=BTCUSDT&start_date=invalid&end_date=invalid"
   ```
   Expected result: An appropriate error message.

3. Test with invalid transaction data:
   ```bash
   curl -X POST "http://localhost:8000/portfolio/transaction" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_TOKEN" -d '{"symbol":"BTCUSDT","amount":-1,"price":50000,"transaction_type":"buy"}'
   ```
   Expected result: An appropriate error message.

## Performance Testing

Test the performance of the API by making multiple requests and measuring the response time:

```bash
time curl -X GET "http://localhost:8000/prediction/latest?symbol=BTCUSDT"
```

Compare the response time with the response time of the original APIs to ensure that the unified API does not introduce significant performance overhead.