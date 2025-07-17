"""
Trade Suggestion Router
----------------------
This module provides endpoints for generating trade suggestions based on prediction signals,
ATR calculations, and user portfolio data.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import pandas as pd
import ta

# Import authentication from crypto_signals
from crypto_signals.src.api import get_current_active_user, User

# Create router
trade_router = APIRouter(prefix="/trade", tags=["Trade Suggestions"])

# Constants for SL and TP multipliers and default risk
SL_MULTIPLIER = 1.0  # Stop Loss = 1 × ATR
TP_MULTIPLIER = 2.0  # Take Profit = 2 × ATR
DEFAULT_RISK_PERCENT = 1.0  # Risk 1% of capital by default

# ========================================================================= #
# Pydantic Models
# ========================================================================= #

class TradeSuggestion(BaseModel):
    """Model for trade suggestion response"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime

# ========================================================================= #
# Utility Functions
# ========================================================================= #

def get_current_price(symbol: str) -> float:
    """
    Get the current price for a symbol.
    Uses the last available price from historical data.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        
    Returns:
        float: Current price
    """
    from crypto_signals.src.data_loader import load_minute
    
    # Load the most recent data
    df = load_minute(symbol, days=1)
    
    # Return the last close price
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for {symbol}")
    
    return float(df["close"].iloc[-1])

def calculate_atr(symbol: str, period: int = 14) -> float:
    """
    Calculate the Average True Range (ATR) for a symbol.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        period: Period for ATR calculation (default: 14)
        
    Returns:
        float: ATR value
    """
    from crypto_signals.src.data_loader import load_minute
    
    # Load historical data (need enough data for the ATR calculation)
    df = load_minute(symbol, days=1)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    
    # Calculate ATR
    atr = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=period)
    
    # Return the last ATR value
    return float(atr.iloc[-1])

def get_user_capital(user: User) -> float:
    """
    Get the user's available capital from their portfolio.
    
    Args:
        user: User object
        
    Returns:
        float: User's available capital
    """
    from crypto_api.api import user_portfolios, Portfolio
    
    # Get user's portfolio
    if user.username not in user_portfolios:
        user_portfolios[user.username] = Portfolio(assets=[], total_value_usd=0)
    
    portfolio = user_portfolios[user.username]
    
    # Return total portfolio value
    return portfolio.total_value_usd

# ========================================================================= #
# Endpoints
# ========================================================================= #

@trade_router.get("/suggest", response_model=TradeSuggestion)
async def suggest_trade(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    risk_percent: float = Query(DEFAULT_RISK_PERCENT, description="Risk percentage (0-100)"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate a trade suggestion based on prediction signals and ATR calculations.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        risk_percent: Percentage of capital to risk (default: 1%)
        current_user: Current authenticated user
        
    Returns:
        TradeSuggestion: Trade suggestion details
    """
    # Get prediction from prediction endpoint
    from crypto_signals.src.predict import predict
    prediction = predict(symbol)
    
    # Check if signal is FLAT
    if prediction["signal"] == "FLAT":
        raise HTTPException(status_code=400, detail="No trade suggestion available (FLAT signal)")
    
    # Get current price
    entry_price = get_current_price(symbol)
    
    # Calculate ATR
    atr = calculate_atr(symbol)
    
    # Calculate stop loss and take profit based on ATR
    if prediction["signal"] == "LONG":
        stop_loss = entry_price - (atr * SL_MULTIPLIER)
        take_profit = entry_price + (atr * TP_MULTIPLIER)
        side = "LONG"
    else:  # SHORT
        stop_loss = entry_price + (atr * SL_MULTIPLIER)
        take_profit = entry_price - (atr * TP_MULTIPLIER)
        side = "SHORT"
    
    # Get user capital
    user_capital = get_user_capital(current_user)
    
    # Calculate position size based on risk
    risk_amount = user_capital * (risk_percent / 100)
    
    # Calculate position size (risk amount / distance to stop loss)
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance == 0:
        raise HTTPException(status_code=400, detail="Cannot calculate position size (zero stop loss distance)")
    
    position_size = risk_amount / sl_distance
    
    # Create trade suggestion
    trade_suggestion = TradeSuggestion(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size,
        timestamp=datetime.utcnow()
    )
    
    return trade_suggestion