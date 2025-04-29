import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time
from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_sma, calculate_ema, calculate_vwap, calculate_obv

class BacktestEngine:
    def __init__(self, symbol, timeframe, period_days, initial_capital, leverage=5, 
                 position_size_pct=5.0, stop_loss_pct=3.0, take_profit_pct=10.0, fee_pct=0.06):
        """
        Initialize the backtesting engine.
        
        Args:
            symbol (str): Trading symbol (e.g., "BTC/USDT:USDT")
            timeframe (str): Candle timeframe (e.g., "1h", "4h", "1d")
            period_days (int): Number of days to backtest
            initial_capital (float): Initial capital in USDT
            leverage (int): Trading leverage
            position_size_pct (float): Position size as percentage of capital (1-100)
            stop_loss_pct (float): Stop loss percentage (e.g., 3 for 3%)
            take_profit_pct (float): Take profit percentage (e.g., 10 for 10%)
            fee_pct (float): Trading fee percentage (e.g., 0.06 for 0.06%, Bitget standard taker fee)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.period_days = period_days
        self.initial_capital = float(initial_capital)
        self.leverage = int(leverage)
        self.position_size_pct = float(position_size_pct) / 100  # Convert to decimal
        self.stop_loss_pct = float(stop_loss_pct) / 100  # Convert to decimal
        self.take_profit_pct = float(take_profit_pct) / 100  # Convert to decimal
        self.fee_pct = float(fee_pct) / 100  # Convert to decimal
        
        # Strategy parameters with defaults
        self.strategy_params = {
            'rsi': {'enabled': True, 'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9},
            'bb': {'enabled': True, 'period': 20, 'deviation': 2},
            'ema': {'enabled': True, 'fast': 9, 'slow': 21},
            'vwap': {'enabled': True},
            'obv': {'enabled': True}
        }
        
        # Initialize data containers
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.current_position = None
    
    def set_strategy_parameters(self, params):
        """Update strategy parameters."""
        # RSI parameters
        if 'rsiEnabled' in params:
            self.strategy_params['rsi']['enabled'] = params['rsiEnabled']
        if 'rsiPeriod' in params:
            self.strategy_params['rsi']['period'] = int(params['rsiPeriod'])
        if 'rsiOverbought' in params:
            self.strategy_params['rsi']['overbought'] = int(params['rsiOverbought'])
        if 'rsiOversold' in params:
            self.strategy_params['rsi']['oversold'] = int(params['rsiOversold'])
        
        # MACD parameters
        if 'macdEnabled' in params:
            self.strategy_params['macd']['enabled'] = params['macdEnabled']
        if 'macdFast' in params:
            self.strategy_params['macd']['fast'] = int(params['macdFast'])
        if 'macdSlow' in params:
            self.strategy_params['macd']['slow'] = int(params['macdSlow'])
        if 'macdSignal' in params:
            self.strategy_params['macd']['signal'] = int(params['macdSignal'])
        
        # Bollinger Bands parameters
        if 'bbEnabled' in params:
            self.strategy_params['bb']['enabled'] = params['bbEnabled']
        if 'bbPeriod' in params:
            self.strategy_params['bb']['period'] = int(params['bbPeriod'])
        if 'bbDeviation' in params:
            self.strategy_params['bb']['deviation'] = float(params['bbDeviation'])
        
        # EMA parameters
        if 'emaEnabled' in params:
            self.strategy_params['ema']['enabled'] = params['emaEnabled']
        if 'emaFast' in params:
            self.strategy_params['ema']['fast'] = int(params['emaFast'])
        if 'emaSlow' in params:
            self.strategy_params['ema']['slow'] = int(params['emaSlow'])
        
        # VWAP parameter
        if 'vwapEnabled' in params:
            self.strategy_params['vwap']['enabled'] = params['vwapEnabled']
        
        # OBV parameter
        if 'obvEnabled' in params:
            self.strategy_params['obv']['enabled'] = params['obvEnabled']
    
    def load_price_data(self, data):
        """
        Load price data for backtesting.
        
        Args:
            data (list): List of OHLCV candles
        """
        if not data:
            raise ValueError("No price data provided")
        
        # Convert to pandas DataFrame with proper column names
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate indicators
        self._calculate_indicators(df)
        
        self.data = df
        
        # Initialize equity curve with starting capital
        self.equity_curve = [(df.iloc[0]['timestamp'].timestamp() * 1000, self.initial_capital)]
        
        return df
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators."""
        # RSI
        if self.strategy_params['rsi']['enabled']:
            df = calculate_rsi(df, period=self.strategy_params['rsi']['period'])
        
        # MACD
        if self.strategy_params['macd']['enabled']:
            df = calculate_macd(
                df, 
                fast=self.strategy_params['macd']['fast'], 
                slow=self.strategy_params['macd']['slow'], 
                signal=self.strategy_params['macd']['signal']
            )
        
        # Bollinger Bands
        if self.strategy_params['bb']['enabled']:
            df = calculate_bollinger_bands(
                df, 
                period=self.strategy_params['bb']['period'], 
                std_dev=self.strategy_params['bb']['deviation']
            )
        
        # EMA
        if self.strategy_params['ema']['enabled']:
            # Fast EMA
            df = calculate_ema(df, period=self.strategy_params['ema']['fast'])
            # Slow EMA
            df = calculate_ema(df, period=self.strategy_params['ema']['slow'])
        
        # VWAP
        if self.strategy_params['vwap']['enabled']:
            df = calculate_vwap(df)
        
        # OBV
        if self.strategy_params['obv']['enabled']:
            df = calculate_obv(df)
    
    def analyze_candle(self, i):
        """
        Analyze a single candle for trading signals.
        
        Args:
            i (int): Index of the candle to analyze
            
        Returns:
            dict: Analysis results with signals
        """
        if i < 20:  # Need enough data for indicators
            return {'signal': 'neutral', 'bullish_count': 0, 'bearish_count': 0, 'neutral_count': 0}
        
        row = self.data.iloc[i]
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        signals = {}
        
        # RSI
        if self.strategy_params['rsi']['enabled'] and 'rsi' in self.data.columns:
            rsi_value = row['rsi']
            if rsi_value < self.strategy_params['rsi']['oversold']:
                signals['rsi'] = 'buy'
                bullish_count += 1
            elif rsi_value > self.strategy_params['rsi']['overbought']:
                signals['rsi'] = 'sell'
                bearish_count += 1
            else:
                signals['rsi'] = 'neutral'
                neutral_count += 1
        
        # MACD
        if self.strategy_params['macd']['enabled'] and 'macd' in self.data.columns and 'macd_signal' in self.data.columns:
            macd = row['macd']
            macd_signal = row['macd_signal']
            
            # MACD crossover
            prev_macd = self.data.iloc[i-1]['macd']
            prev_macd_signal = self.data.iloc[i-1]['macd_signal']
            
            if macd > macd_signal and prev_macd <= prev_macd_signal:
                signals['macd'] = 'buy'
                bullish_count += 1
            elif macd < macd_signal and prev_macd >= prev_macd_signal:
                signals['macd'] = 'sell'
                bearish_count += 1
            else:
                signals['macd'] = 'neutral'
                neutral_count += 1
        
        # Bollinger Bands
        if self.strategy_params['bb']['enabled'] and 'bb_upper' in self.data.columns and 'bb_lower' in self.data.columns:
            close = row['close']
            bb_upper = row['bb_upper']
            bb_lower = row['bb_lower']
            
            if close < bb_lower:
                signals['bb'] = 'buy'
                bullish_count += 1
            elif close > bb_upper:
                signals['bb'] = 'sell'
                bearish_count += 1
            else:
                signals['bb'] = 'neutral'
                neutral_count += 1
        
        # EMA Crossover
        if self.strategy_params['ema']['enabled']:
            fast_col = f'ema_{self.strategy_params["ema"]["fast"]}'
            slow_col = f'ema_{self.strategy_params["ema"]["slow"]}'
            
            if fast_col in self.data.columns and slow_col in self.data.columns:
                fast_ema = row[fast_col]
                slow_ema = row[slow_col]
                prev_fast_ema = self.data.iloc[i-1][fast_col]
                prev_slow_ema = self.data.iloc[i-1][slow_col]
                
                if fast_ema > slow_ema and prev_fast_ema <= prev_slow_ema:
                    signals['ema'] = 'buy'
                    bullish_count += 1
                elif fast_ema < slow_ema and prev_fast_ema >= prev_slow_ema:
                    signals['ema'] = 'sell'
                    bearish_count += 1
                else:
                    signals['ema'] = 'neutral'
                    neutral_count += 1
        
        # VWAP
        if self.strategy_params['vwap']['enabled'] and 'vwap' in self.data.columns:
            close = row['close']
            vwap = row['vwap']
            
            if close > vwap:
                signals['vwap'] = 'buy'
                bullish_count += 1
            elif close < vwap:
                signals['vwap'] = 'sell'
                bearish_count += 1
            else:
                signals['vwap'] = 'neutral'
                neutral_count += 1
        
        # OBV Trend
        if self.strategy_params['obv']['enabled'] and 'obv' in self.data.columns:
            obv = row['obv']
            obv_5 = self.data.iloc[i-5:i]['obv'].mean()
            
            if obv > obv_5:
                signals['obv'] = 'buy'
                bullish_count += 1
            elif obv < obv_5:
                signals['obv'] = 'sell'
                bearish_count += 1
            else:
                signals['obv'] = 'neutral'
                neutral_count += 1
        
        # Determine overall signal
        if bullish_count > bearish_count + neutral_count:
            overall_signal = 'buy'
        elif bearish_count > bullish_count + neutral_count:
            overall_signal = 'sell'
        else:
            overall_signal = 'neutral'
        
        return {
            'signal': overall_signal,
            'signals': signals,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count
        }
    
    def run_backtest(self):
        """Run the backtest."""
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded. Call load_price_data() first.")
        
        current_capital = self.initial_capital
        max_capital = self.initial_capital
        
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            timestamp = row['timestamp']
            unix_timestamp = timestamp.timestamp() * 1000
            
            # Check if we need to exit existing position
            if self.current_position is not None:
                exit_result = self._check_exit_conditions(row, i)
                
                if exit_result['exit']:
                    # Close the position
                    exit_price = exit_result['price']
                    exit_type = exit_result['reason']
                    
                    trade_result = self._close_position(exit_price, unix_timestamp, exit_type)
                    current_capital = trade_result['new_capital']
                    
                    # Update max capital for drawdown calculation
                    if current_capital > max_capital:
                        max_capital = current_capital
                    
                    # Update equity curve
                    self.equity_curve.append((unix_timestamp, current_capital))
            
            # Check for new entry signals if we're not in a position
            if self.current_position is None:
                analysis = self.analyze_candle(i)
                
                if analysis['signal'] == 'buy':
                    # Open long position
                    entry_price = row['close']
                    position_size = current_capital * self.position_size_pct * self.leverage
                    stop_loss = entry_price * (1 - self.stop_loss_pct)
                    take_profit = entry_price * (1 + self.take_profit_pct)
                    
                    # Calculate fees
                    fee = position_size * self.fee_pct
                    
                    self.current_position = {
                        'type': 'long',
                        'entry_price': entry_price,
                        'entry_time': unix_timestamp,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'fee': fee
                    }
                    
                elif analysis['signal'] == 'sell':
                    # Open short position
                    entry_price = row['close']
                    position_size = current_capital * self.position_size_pct * self.leverage
                    stop_loss = entry_price * (1 + self.stop_loss_pct)
                    take_profit = entry_price * (1 - self.take_profit_pct)
                    
                    # Calculate fees
                    fee = position_size * self.fee_pct
                    
                    self.current_position = {
                        'type': 'short',
                        'entry_price': entry_price,
                        'entry_time': unix_timestamp,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'fee': fee
                    }
        
        # Close any remaining position at the end
        if self.current_position is not None:
            last_row = self.data.iloc[-1]
            exit_price = last_row['close']
            unix_timestamp = last_row['timestamp'].timestamp() * 1000
            
            trade_result = self._close_position(exit_price, unix_timestamp, 'end_of_test')
            current_capital = trade_result['new_capital']
            
            # Update equity curve
            self.equity_curve.append((unix_timestamp, current_capital))
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': current_capital,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics
        }
    
    def _check_exit_conditions(self, row, index):
        """
        Check if we should exit the current position.
        
        Args:
            row (Series): Current candle data
            index (int): Index of the current candle
            
        Returns:
            dict: Exit results
        """
        if self.current_position is None:
            return {'exit': False}
        
        high = row['high']
        low = row['low']
        close = row['close']
        
        if self.current_position['type'] == 'long':
            # Check stop loss
            if low <= self.current_position['stop_loss']:
                return {'exit': True, 'price': self.current_position['stop_loss'], 'reason': 'stop_loss'}
            
            # Check take profit
            if high >= self.current_position['take_profit']:
                return {'exit': True, 'price': self.current_position['take_profit'], 'reason': 'take_profit'}
            
            # Check for trend reversal
            if index > 0:
                analysis = self.analyze_candle(index)
                if analysis['signal'] == 'sell':
                    return {'exit': True, 'price': close, 'reason': 'signal_reversal'}
        
        elif self.current_position['type'] == 'short':
            # Check stop loss
            if high >= self.current_position['stop_loss']:
                return {'exit': True, 'price': self.current_position['stop_loss'], 'reason': 'stop_loss'}
            
            # Check take profit
            if low <= self.current_position['take_profit']:
                return {'exit': True, 'price': self.current_position['take_profit'], 'reason': 'take_profit'}
            
            # Check for trend reversal
            if index > 0:
                analysis = self.analyze_candle(index)
                if analysis['signal'] == 'buy':
                    return {'exit': True, 'price': close, 'reason': 'signal_reversal'}
        
        return {'exit': False}
    
    def _close_position(self, exit_price, exit_time, exit_reason):
        """
        Close the current position and calculate results.
        
        Args:
            exit_price (float): Exit price
            exit_time (int): Exit timestamp
            exit_reason (str): Reason for exit
            
        Returns:
            dict: Trade results including new capital
        """
        if self.current_position is None:
            return {'new_capital': self.equity_curve[-1][1]}
        
        # Calculate P&L
        entry_price = self.current_position['entry_price']
        position_size = self.current_position['position_size']
        fee = self.current_position['fee']
        
        if self.current_position['type'] == 'long':
            profit_loss = position_size * ((exit_price / entry_price) - 1)
        else:  # short
            profit_loss = position_size * ((entry_price / exit_price) - 1)
        
        # Account for fees
        profit_loss -= fee
        profit_loss -= (position_size * self.fee_pct)  # Exit fee
        
        # Calculate P&L percentage relative to the position size
        profit_loss_percent = profit_loss / position_size
        
        # Calculate new capital
        current_capital = self.equity_curve[-1][1]
        new_capital = current_capital + profit_loss
        
        # Record trade
        trade = {
            'type': self.current_position['type'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': self.current_position['entry_time'],
            'exit_time': exit_time,
            'position_size': position_size,
            'profit_loss': profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade)
        
        # Reset current position
        self.current_position = None
        
        return {'new_capital': new_capital, 'trade': trade}
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'profit_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_profit': 0,
                'avg_loss': 0
            }
        
        # Total trades
        total_trades = len(self.trades)
        
        # Winning trades
        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        win_count = len(winning_trades)
        
        # Win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Total return
        initial_capital = self.initial_capital
        final_capital = self.equity_curve[-1][1] if self.equity_curve else initial_capital
        total_return = (final_capital - initial_capital) / initial_capital
        profit_loss = final_capital - initial_capital
        
        # Calculate max drawdown
        max_drawdown = 0
        peak_capital = initial_capital
        
        for timestamp, capital in self.equity_curve:
            if capital > peak_capital:
                peak_capital = capital
            
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        
        for i in range(1, len(self.equity_curve)):
            prev_capital = self.equity_curve[i-1][1]
            curr_capital = self.equity_curve[i][1]
            
            if prev_capital > 0:
                returns.append((curr_capital - prev_capital) / prev_capital)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Average profit and loss
        profit_trades = [t['profit_loss'] for t in self.trades if t['profit_loss'] > 0]
        loss_trades = [t['profit_loss'] for t in self.trades if t['profit_loss'] <= 0]
        
        avg_profit = np.mean(profit_trades) if profit_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_loss': profit_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss
        }