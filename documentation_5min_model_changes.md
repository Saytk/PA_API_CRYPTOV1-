# 5-Minute Model Visualization Changes

## Overview
This document describes the changes made to implement a separate visualization for the 5-minute prediction model in the Quantia application. The issue was that both the main prediction visualization and the 5-minute model visualization were using the same prediction data, which made it difficult to distinguish between them.

## Changes Made

### 1. Added a New Endpoint in PredictionController.cs
Created a new endpoint `Get5MinutePredictions` that specifically fetches predictions from the 5-minute aggregated model:

```csharp
// POST /Prediction/Get5MinutePredictions
[HttpPost("Get5MinutePredictions")]
public async Task<IActionResult> Get5MinutePredictions([FromBody] PredictRequest dto)
{
    try
    {
        string url;

        // Check if we're requesting historical data
        if (dto.Date.HasValue || dto.Days.HasValue)
        {
            // Use the historical endpoint with the use-aggregated-model parameter set to true
            int days = dto.Days ?? 7; // Default to 7 days if not specified
            url = $"{ML_API_BASE}/prediction/historical/{dto.Symbol}?days={days}&use_aggregated=true";
        }
        else
        {
            // Use the pattern endpoint for latest predictions with the use-aggregated-model parameter
            // First, ensure we're using the aggregated model
            await MlClient.PostAsync($"{ML_API_BASE}/prediction/use-aggregated-model?use_aggregated=true", null);
            
            // Then get the prediction
            url = $"{dto.ApiUrl}?symbol={dto.Symbol}&use_aggregated=true";
        }

        var json = await MlClient.GetStringAsync(url);
        return Content(json, "application/json");
    }
    catch (Exception ex)
    {
        return Problem($"Error fetching 5-minute predictions: {ex.Message}");
    }
}
```

This endpoint ensures that the `use_aggregated` parameter is set to true, which tells the Python backend to use the aggregated 5-minute model for predictions.

### 2. Added a Function to Fetch 5-Minute Predictions in Index.cshtml
Added a new JavaScript function `fetch5MinutePredictions()` that calls the new endpoint:

```javascript
// Function to fetch 5-minute model predictions
function fetch5MinutePredictions() {
    const apiUrl = document.getElementById('api-url').value.trim();
    const symbol = document.getElementById('symbol').value.trim();
    
    fetch('/Prediction/Get5MinutePredictions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiUrl, symbol })
    })
    .then(r => r.json())
    .then(data => create5MinuteVisualization(data))
    .catch(err => console.error('Error fetching 5-minute predictions:', err));
}
```

### 3. Modified the displayPredictions Function
Updated the `displayPredictions` function to call `fetch5MinutePredictions()` instead of directly passing the same data to the 5-minute visualization:

```javascript
// Update the displayPredictions function to also fetch 5-minute predictions
const originalDisplayPredictions = displayPredictions;
displayPredictions = function(data) {
    originalDisplayPredictions(data);
    fetch5MinutePredictions();
};
```

### 4. Enhanced the 5-Minute Visualization
Significantly improved the `create5MinuteVisualization` function to better visualize the 5-minute model predictions:

- Added calculation of predicted price changes based on the prediction probabilities
- Created predicted prices by applying these changes to the current prices
- Changed the visualization to show:
  - Current prices as a line and small gray markers
  - Diamond-shaped markers for the 5-minute predicted prices
  - Color coding based on prediction confidence (green for up, red for down)
  - Enhanced hover text showing direction (up/down arrow), confidence percentage, and probability
- Updated the chart title to indicate it's using the aggregated model

## How It Works

1. When the user loads the page or clicks the "Fetch Predictions" button, the application fetches regular predictions and displays them in the main chart.
2. After displaying the regular predictions, it then fetches 5-minute model predictions using the new endpoint.
3. The 5-minute model predictions are displayed in a separate chart below the main chart.
4. The 5-minute visualization shows diamond-shaped markers that represent the predicted prices after 5 minutes, based on the model's predictions.
5. The color of each marker indicates the confidence of the prediction (green for up, red for down).
6. Hovering over a marker shows detailed information about the prediction, including the direction, confidence percentage, and probability.

## Technical Details

The 5-minute model is an aggregated model that predicts price movements over a 5-minute period. It's implemented in the Python backend in the `crypto_signals_compat.py` file. The `USE_AGGREGATED_MODEL` flag determines whether to use this model or individual horizon models.

The changes made ensure that the 5-minute visualization uses the aggregated model, while the main visualization continues to use the regular predictions.