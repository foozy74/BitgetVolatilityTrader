import numpy as np
import pandas as pd
import logging
import json
import time
from datetime import datetime
import math
from indicators import analyze_coin
from backtesting import BacktestEngine

# Setup logging
logger = logging.getLogger('trading_bot')

class AdaptiveStrategy:
    """
    Adaptive trading strategy that dynamically adjusts signal requirements
    based on market conditions and historical performance.
    """
    def __init__(self):
        self.signal_weights = {
            'rsi': 1.0,
            'macd': 1.0,
            'obv': 1.0,
            'vwap': 1.0,
            'sma': 1.0,
            'bollinger_bands': 1.0
        }
        self.signal_success_rate = {}
        self.trade_history = {}
        self.volatility_cache = {}
        self.last_volatility_update = {}
        # How often to update volatility (in seconds)
        self.volatility_update_interval = 3600  # 1 hour

    def calculate_market_volatility(self, df, symbol):
        """
        Calculate the current market volatility for a symbol.
        Returns volatility as a normalized value between 0 and 1.
        Higher values indicate higher volatility.
        """
        # Check if we have recent volatility data
        current_time = time.time()
        if (symbol in self.volatility_cache and 
            current_time - self.last_volatility_update.get(symbol, 0) < self.volatility_update_interval):
            return self.volatility_cache[symbol]
        
        try:
            # Calculate volatility as normalized ATR (Average True Range)
            df = df.copy()
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            # Calculate true range
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calculate ATR-14
            df['atr'] = df['true_range'].rolling(window=14).mean()
            
            # Normalize ATR as percentage of price
            if len(df) > 0 and df['close'].iloc[-1] > 0:
                recent_atr = df['atr'].iloc[-1]
                recent_price = df['close'].iloc[-1]
                volatility = min(1.0, max(0.1, recent_atr / recent_price))
                
                # Cache the result
                self.volatility_cache[symbol] = volatility
                self.last_volatility_update[symbol] = current_time
                
                return volatility
            
            return 0.5  # Default medium volatility if calculation fails
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0.5  # Default medium volatility
    
    def get_required_signals(self, volatility):
        """
        Determine how many signals are required based on current volatility.
        Higher volatility requires more confirming signals.
        
        Returns: Float representing required weighted signal strength
        """
        # Less restrictive thresholds for more trading opportunities
        if volatility >= 0.8:
            return 3.0  # Very high volatility - need moderate confirmation (was 5)
        elif volatility >= 0.6:
            return 2.5  # High volatility (was 4)
        elif volatility >= 0.4:
            return 2.0  # Medium volatility (was 3)
        elif volatility >= 0.2:
            return 1.5  # Low volatility (was 2)
        else:
            return 1.0  # Very low volatility - can be very aggressive (was 2)
    
    def update_signal_weights(self, symbol, analysis, trade_result=None):
        """
        Update the weight of each signal based on its historical performance.
        If trade_result is provided, use it to update success rates.
        """
        # Initialize all weights to be higher by default (more aggressive trading)
        # If no learning data yet, start with higher weights to encourage more trades
        for key in self.signal_weights:
            if self.signal_weights[key] < 1.0:
                self.signal_weights[key] = 1.0
        
        if trade_result is not None:
            # First time seeing this symbol
            if symbol not in self.signal_success_rate:
                self.signal_success_rate[symbol] = {
                    'rsi': {'correct': 0, 'total': 0},
                    'macd': {'correct': 0, 'total': 0},
                    'obv': {'correct': 0, 'total': 0},
                    'vwap': {'correct': 0, 'total': 0},
                    'sma': {'correct': 0, 'total': 0},
                    'bollinger_bands': {'correct': 0, 'total': 0}
                }
            
            # Update success rates for each indicator
            for indicator, data in analysis.items():
                if 'signal' in data:
                    indicator_signal = data['signal']
                    # Skip neutral signals
                    if indicator_signal == 'neutral':
                        continue
                    
                    # Evaluate if the signal was correct
                    signal_correct = False
                    if trade_result['profitable']:
                        # Signal was correct if it matched the trade direction
                        if (trade_result['direction'] == 'long' and indicator_signal == 'buy') or \
                           (trade_result['direction'] == 'short' and indicator_signal == 'sell'):
                            signal_correct = True
                    else:
                        # Signal was wrong if it matched the trade direction
                        if (trade_result['direction'] == 'long' and indicator_signal == 'buy') or \
                           (trade_result['direction'] == 'short' and indicator_signal == 'sell'):
                            signal_correct = False
                        else:
                            signal_correct = True
                    
                    # Update indicator success rate
                    if signal_correct:
                        self.signal_success_rate[symbol][indicator]['correct'] += 1
                    self.signal_success_rate[symbol][indicator]['total'] += 1
        
        # Update weights based on success rates
        if symbol in self.signal_success_rate:
            for indicator, stats in self.signal_success_rate[symbol].items():
                if stats['total'] > 0:
                    success_rate = stats['correct'] / stats['total']
                    # Apply sigmoid function to smooth weights between 0.5 and 1.5
                    # Adjusted to be more aggressive - we want higher weights
                    weight = 0.75 + 1.0 / (1.0 + math.exp(-5 * (success_rate - 0.5)))
                    self.signal_weights[indicator] = weight
    
    def should_enter_trade(self, df, symbol, analysis=None):
        """
        Determine if we should enter a trade based on adaptive strategy.
        
        Args:
            df: Price and indicator DataFrame
            symbol: Trading symbol
            analysis: Pre-calculated analysis (optional)
            
        Returns:
            tuple: (should_trade, direction)
        """
        if analysis is None:
            analysis = analyze_coin(df)
        
        # Calculate current market volatility
        volatility = self.calculate_market_volatility(df, symbol)
        
        # Determine required number of signals based on volatility
        required_signals = self.get_required_signals(volatility)
        
        # Count weighted signals
        weighted_bullish = 0
        weighted_bearish = 0
        
        # Helper function to safely get signals
        def safe_get_signal(signals_dict, key, subkey='signal', default='neutral'):
            try:
                if key in signals_dict and isinstance(signals_dict[key], dict):
                    if subkey in signals_dict[key]:
                        return str(signals_dict[key][subkey])
                return default
            except:
                return default
        
        # Process each indicator
        for indicator, weight in self.signal_weights.items():
            signal = safe_get_signal(analysis, indicator)
            
            if signal == 'buy':
                weighted_bullish += weight
            elif signal == 'sell':
                weighted_bearish += weight
        
        # Log detailed analysis
        logger.debug(f"{symbol} adaptive strategy: volatility={volatility:.2f}, required_signals={required_signals}")
        logger.debug(f"{symbol} weighted signals: bullish={weighted_bullish:.2f}, bearish={weighted_bearish:.2f}")
        
        # Make trading decision
        if weighted_bullish >= required_signals and weighted_bullish > weighted_bearish:
            return True, 'long'
        elif weighted_bearish >= required_signals and weighted_bearish > weighted_bullish:
            return True, 'short'
        else:
            return False, None
    
    def record_trade_result(self, symbol, analysis, direction, entry_price, exit_price, exit_reason):
        """
        Record the result of a completed trade to update signal weights.
        
        Args:
            symbol: Trading symbol
            analysis: Analysis at trade entry
            direction: 'long' or 'short'
            entry_price: Entry price
            exit_price: Exit price
            exit_reason: Reason for exit ('stop_loss', 'take_profit', 'manual')
        """
        # Calculate if trade was profitable
        profitable = False
        if direction == 'long':
            profitable = exit_price > entry_price
        else:  # short
            profitable = exit_price < entry_price
        
        # Record trade
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        
        trade_result = {
            'timestamp': datetime.now().timestamp(),
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'profitable': profitable,
            'profit_pct': (exit_price - entry_price) / entry_price * 100 if direction == 'long' else 
                         (entry_price - exit_price) / entry_price * 100
        }
        
        self.trade_history[symbol].append(trade_result)
        
        # Update signal weights
        self.update_signal_weights(symbol, analysis, trade_result)
        
        return trade_result

# Global instance for use in main.py
adaptive_strategy = AdaptiveStrategy()