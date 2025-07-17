import uvicorn
import os
import sys
from pathlib import Path

def main():
    """
    Run the FastAPI server for crypto predictions.
    This script starts the API server that the dashboard connects to.
    """
    print("Starting Crypto Forecast API Server...")
    
    # Add the parent directory to the path so that imports work correctly
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "crypto_forecast_ml.predictor.serve_api:app",
        host="0.0.0.0",
        port=8009,
        reload=True
    )

if __name__ == "__main__":
    main()