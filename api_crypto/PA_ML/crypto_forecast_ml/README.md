# Crypto Forecast ML

A machine learning project for predicting cryptocurrency (BTC and ETH) price movements using historical OHLCV data from BigQuery.

## Project Structure

- **BigQuery Integration**
  - `data_loader.py`: Loads BTC/ETH 1-minute OHLCV data from BigQuery
  - `config/credentials.json`: GCP service account credentials

- **Feature Engineering**
  - `features/technical_indicators.py`: Computes technical indicators (SMA, EMA, RSI, MACD, etc.)
  - `features/target_builder.py`: Generates prediction targets like next_close or direction

- **Model Training & Evaluation**
  - `training/train_model.py`: Trains XGBoost or similar models
  - `training/evaluate_model.py`: Computes evaluation metrics
  - `models/`: Where trained model files (e.g., `.pkl`) will be saved

- **Prediction & Inference**
  - `predictor/predict.py`: Loads the model and makes predictions
  - `predictor/serve_api.py`: FastAPI endpoint to expose predictions

- **Notebook**
  - `notebooks/eda.ipynb`: Used for exploratory analysis and visualization

- **Utilities**
  - `utils/logger.py`: Basic logging configuration
  - `requirements.txt`: Dependencies list

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Place your Google Cloud credentials in `config/credentials.json`

## Usage

The implementation details will be filled in by ChatGPT. This project structure serves as a scaffold for building a complete cryptocurrency forecasting system.

## License

[MIT License](LICENSE)