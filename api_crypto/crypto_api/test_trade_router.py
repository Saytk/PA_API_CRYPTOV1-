"""
Unit tests for the trade suggestion functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
from datetime import datetime

# Import the app and router
from crypto_api.api import app
import crypto_api.trade_router
from crypto_api.trade_router import (
    get_current_price,
    calculate_atr,
    get_user_capital,
    TradeSuggestion
)

# Direct import of suggest_trade function
from crypto_api.trade_router import suggest_trade

# Create test client
client = TestClient(app)

# Mock user for testing
mock_user = MagicMock()
mock_user.username = "test_user"

class TestTradeSuggestionUtils(unittest.TestCase):
    """Test utility functions for trade suggestions."""

    def test_module_attributes(self):
        """Test to print module attributes for debugging."""
        import crypto_api.trade_router
        print("\nModule attributes:", dir(crypto_api.trade_router))
        print("\nHas suggest_trade:", hasattr(crypto_api.trade_router, "suggest_trade"))

    @patch("crypto_signals.src.data_loader.load_minute")
    def test_get_current_price(self, mock_load_minute):
        """Test getting current price."""
        # Mock data
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__getitem__.return_value.iloc.__getitem__.return_value = 50000.0
        mock_load_minute.return_value = mock_df

        # Test function
        price = get_current_price("BTCUSDT")

        # Assertions
        self.assertEqual(price, 50000.0)
        mock_load_minute.assert_called_once_with("BTCUSDT", days=1)

    @patch("crypto_signals.src.data_loader.load_minute")
    def test_get_current_price_empty_data(self, mock_load_minute):
        """Test getting current price with empty data."""
        # Mock empty data
        mock_df = MagicMock()
        mock_df.empty = True
        mock_load_minute.return_value = mock_df

        # Test function should raise exception
        with self.assertRaises(HTTPException) as context:
            get_current_price("UNKNOWN")

        # Check exception details
        self.assertEqual(context.exception.status_code, 404)
        self.assertEqual(context.exception.detail, "No price data found for UNKNOWN")

    @patch("crypto_signals.src.data_loader.load_minute")
    @patch("ta.volatility.average_true_range")
    def test_calculate_atr(self, mock_atr, mock_load_minute):
        """Test calculating ATR."""
        # Mock data
        mock_df = MagicMock()
        mock_df.empty = False
        mock_atr_result = MagicMock()
        mock_atr_result.iloc.__getitem__.return_value = 1500.0
        mock_atr.return_value = mock_atr_result
        mock_load_minute.return_value = mock_df

        # Test function
        atr = calculate_atr("BTCUSDT")

        # Assertions
        self.assertEqual(atr, 1500.0)
        mock_load_minute.assert_called_once_with("BTCUSDT", days=1)
        mock_atr.assert_called_once()

    @patch("crypto_api.api.user_portfolios")
    def test_get_user_capital(self, mock_user_portfolios):
        """Test getting user capital."""
        # Mock portfolio
        mock_portfolio_instance = MagicMock()
        mock_portfolio_instance.total_value_usd = 10000.0

        # Mock user portfolios dictionary access
        mock_user_portfolios.__getitem__.return_value = mock_portfolio_instance

        # Test function
        capital = get_user_capital(mock_user)

        # Assertions
        self.assertEqual(capital, 10000.0)

class TestTradeSuggestionEndpoint:
    """Test the trade suggestion endpoint."""

    @pytest.mark.asyncio
    @patch("crypto_signals.src.predict.predict")
    @patch("crypto_api.trade_router.get_current_price")
    @patch("crypto_api.trade_router.calculate_atr")
    @patch("crypto_api.trade_router.get_user_capital")
    async def test_suggest_trade_long(self, mock_get_capital, mock_calc_atr, mock_get_price, mock_predict):
        """Test suggesting a LONG trade."""
        # Mock data
        mock_predict.return_value = {
            "symbol": "BTCUSDT",
            "timestamp": "2023-01-01T00:00:00",
            "prob_up": 0.75,
            "signal": "LONG",
            "confidence": 0.5,
            "using_incomplete_candle": True
        }
        mock_get_price.return_value = 50000.0
        mock_calc_atr.return_value = 1500.0
        mock_get_capital.return_value = 10000.0

        # Test function
        result = await suggest_trade("BTCUSDT", 1.0, mock_user)

        # Assertions
        assert result.symbol == "BTCUSDT"
        assert result.side == "LONG"
        assert result.entry_price == 50000.0
        assert result.stop_loss == 48500.0  # 50000 - (1500 * 1.0)
        assert result.take_profit == 53000.0  # 50000 + (1500 * 2.0)
        assert result.position_size == 100.0 / 1500.0  # (10000 * 0.01) / 1500

    @pytest.mark.asyncio
    @patch("crypto_signals.src.predict.predict")
    @patch("crypto_api.trade_router.get_current_price")
    @patch("crypto_api.trade_router.calculate_atr")
    @patch("crypto_api.trade_router.get_user_capital")
    async def test_suggest_trade_short(self, mock_get_capital, mock_calc_atr, mock_get_price, mock_predict):
        """Test suggesting a SHORT trade."""
        # Mock data
        mock_predict.return_value = {
            "symbol": "BTCUSDT",
            "timestamp": "2023-01-01T00:00:00",
            "prob_up": 0.25,
            "signal": "SHORT",
            "confidence": 0.5,
            "using_incomplete_candle": True
        }
        mock_get_price.return_value = 50000.0
        mock_calc_atr.return_value = 1500.0
        mock_get_capital.return_value = 10000.0

        # Test function
        result = await suggest_trade("BTCUSDT", 1.0, mock_user)

        # Assertions
        assert result.symbol == "BTCUSDT"
        assert result.side == "SHORT"
        assert result.entry_price == 50000.0
        assert result.stop_loss == 51500.0  # 50000 + (1500 * 1.0)
        assert result.take_profit == 47000.0  # 50000 - (1500 * 2.0)
        assert result.position_size == 100.0 / 1500.0  # (10000 * 0.01) / 1500

    @pytest.mark.asyncio
    @patch("crypto_signals.src.predict.predict")
    async def test_suggest_trade_flat(self, mock_predict):
        """Test suggesting a trade with FLAT signal."""
        # Mock data
        mock_predict.return_value = {
            "symbol": "BTCUSDT",
            "timestamp": "2023-01-01T00:00:00",
            "prob_up": 0.5,
            "signal": "FLAT",
            "confidence": 0.0,
            "using_incomplete_candle": True
        }

        # Test function should raise exception
        with pytest.raises(HTTPException) as excinfo:
            await suggest_trade("BTCUSDT", 1.0, mock_user)

        # Check exception details
        assert excinfo.value.status_code == 400
        assert excinfo.value.detail == "No trade suggestion available (FLAT signal)"

    @pytest.mark.asyncio
    @patch("crypto_signals.src.predict.predict")
    @patch("crypto_api.trade_router.get_current_price")
    @patch("crypto_api.trade_router.calculate_atr")
    @patch("crypto_api.trade_router.get_user_capital")
    async def test_suggest_trade_no_capital(self, mock_get_capital, mock_calc_atr, mock_get_price, mock_predict):
        """Test suggesting a trade with no capital."""
        # Mock data
        mock_predict.return_value = {
            "symbol": "BTCUSDT",
            "timestamp": "2023-01-01T00:00:00",
            "prob_up": 0.75,
            "signal": "LONG",
            "confidence": 0.5,
            "using_incomplete_candle": True
        }
        mock_get_price.return_value = 50000.0
        mock_calc_atr.return_value = 1500.0
        mock_get_capital.return_value = 0.0

        # Test function
        result = await suggest_trade("BTCUSDT", 1.0, mock_user)

        # Assertions
        assert result.position_size == 0.0  # No capital means no position

if __name__ == "__main__":
    unittest.main()
