"""
Test script for the trade suggestion functionality.
This script tests both the direct function call and the API endpoint.
"""

import requests
import json
from crypto_signals.src.predict import predict
from crypto_signals.src.train_tradesuggestion import suggest_trade

def test_direct_function():
    """Test the trade suggestion function directly."""
    print("Testing direct function call...")
    
    # Get prediction for BTCUSDT
    prediction = predict("BTCUSDT")
    
    # Print the prediction and trade suggestion
    print(f"Prediction: {json.dumps(prediction, indent=2)}")
    
    # Get trade suggestion directly
    trade_suggestion = suggest_trade("BTCUSDT", prediction["prob_up"])
    print(f"Direct trade suggestion: {json.dumps(trade_suggestion, indent=2)}")
    
    # Compare with the trade suggestion from the prediction
    print(f"Are they equal? {trade_suggestion == prediction['trade_suggestion']}")

def test_api_endpoint():
    """Test the trade suggestion API endpoint."""
    print("\nTesting API endpoint...")
    
    # The API should be running for this test to work
    try:
        # Get prediction from the API
        response = requests.get("http://localhost:8000/prediction/latest?symbol=BTCUSDT")
        prediction = response.json()
        print(f"API prediction: {json.dumps(prediction, indent=2)}")
        
        # Get trade suggestion from the API
        response = requests.get("http://localhost:8000/prediction/trade-suggestion?symbol=BTCUSDT")
        trade_suggestion = response.json()
        print(f"API trade suggestion: {json.dumps(trade_suggestion, indent=2)}")
        
        # Compare with the trade suggestion from the prediction
        print(f"Are they equal? {trade_suggestion == prediction['trade_suggestion']}")
    except requests.exceptions.ConnectionError:
        print("API not running. Start the API with 'python -m crypto_api.run_api' first.")

if __name__ == "__main__":
    test_direct_function()
    test_api_endpoint()