#!/usr/bin/env python3
"""
nvdia 101 - Momentum, trend_following, risk_adjusted_momentum Trading Strategy

Strategy Type: momentum, trend_following, risk_adjusted_momentum
Description: nvdia 101
Created: 2025-07-08T15:18:28.699Z

WARNING: This is a template implementation. Thoroughly backtest before live trading.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class nvdia101Strategy:
    """
    nvdia 101 Implementation
    
    Strategy Type: momentum, trend_following, risk_adjusted_momentum
    Risk Level: Monitor drawdowns and position sizes carefully
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.positions = {}
        self.performance_metrics = {}
        logger.info(f"Initialized nvdia 101 strategy")
        
    def get_default_config(self):
        """Default configuration parameters"""
        return {
            'max_position_size': 0.05,  # 5% max position size
            'stop_loss_pct': 0.05,      # 5% stop loss
            'lookback_period': 20,       # 20-day lookback
            'rebalance_freq': 'daily',   # Rebalancing frequency
            'transaction_costs': 0.001,  # 0.1% transaction costs
        }
    
    def load_data(self, symbols, start_date, end_date):
        """Load market data for analysis"""
        try:
            import yfinance as yf
            data = yf.download(symbols, start=start_date, end=end_date)
            logger.info(f"Loaded data for {len(symbols)} symbols")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

# =============================================================================
# USER'S STRATEGY IMPLEMENTATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MomentumStrategy:
    def __init__(self, data, short_window=20, long_window=50, risk_free_rate=0.01):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.risk_free_rate = risk_free_rate
        self.signals = None
        self.positions = None

    def generate_signals(self):
        self.data['short_mavg'] = self.data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        self.data['long_mavg'] = self.data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        self.data['signal'] = 0
        self.data['signal'][self.short_window:] = np.where(self.data['short_mavg'][self.short_window:] > self.data['long_mavg'][self.short_window:], 1, 0)
        self.data['positions'] = self.data['signal'].diff()
        self.signals = self.data[['Close', 'short_mavg', 'long_mavg', 'signal', 'positions']]

    def backtest_strategy(self):
        self.data['strategy_returns'] = self.data['Close'].pct_change() * self.data['signal'].shift(1)
        self.data['cumulative_strategy_returns'] = (1 + self.data['strategy_returns']).cumprod()
        self.data['cumulative_market_returns'] = (1 + self.data['Close'].pct_change()).cumprod()
        self.calculate_performance_metrics()

    def calculate_performance_metrics(self):
        self.sharpe_ratio = (self.data['strategy_returns'].mean() - self.risk_free_rate) / self.data['strategy_returns'].std()
        self.max_drawdown = (self.data['cumulative_strategy_returns'].max() - self.data['cumulative_strategy_returns']).max()
        print("Sharpe Ratio: %.2f" % self.sharpe_ratio)
        print("Max Drawdown: %.2f" % self.max_drawdown)

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['cumulative_strategy_returns'], label='Strategy Returns')
        plt.plot(self.data['cumulative_market_returns'], label='Market Returns')
        plt.title('Momentum Strategy vs Market Returns')
        plt.legend()
        plt.show()

def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
    prices = np.random.normal(loc=0.001, scale=0.02, size=len(dates)).cumsum() + 100
    data = pd.DataFrame(data=prices, index=dates, columns=['Close'])
    return data

if __name__ == "__main__":
    try:
        sample_data = generate_sample_data()
        strategy = MomentumStrategy(data=sample_data)
        strategy.generate_signals()
        strategy.backtest_strategy()
        strategy.plot_results()
    except Exception as e:
        print("Error occurred: %s" % str(e))

# =============================================================================
# STRATEGY EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    strategy = nvdia101Strategy()
    print(f"Strategy '{strategyName}' initialized successfully!")
    
    # Example data loading
    symbols = ['SPY', 'QQQ', 'IWM']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"Loading data for symbols: {symbols}")
    data = strategy.load_data(symbols, start_date, end_date)
    
    if data is not None:
        print(f"Data loaded successfully. Shape: {data.shape}")
        print("Strategy ready for backtesting!")
    else:
        print("Failed to load data. Check your internet connection.")
