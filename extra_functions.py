import time
import logging
import pandas as pd
import datetime

# Set up logger
logger = logging.getLogger('trading_bot')

# Function to update coin performance data for dashboard
async def update_coin_performance(self, coins_to_analyze):
    """
    Update coin performance data for the dashboard.
    
    Args:
        coins_to_analyze (list): List of coins to gather performance data
    """
    try:
        from main import shared_data, DEBUG_MODE
        
        coin_performance = {}
        
        for symbol in coins_to_analyze:
            try:
                # Get candle data for the last 24 hours
                candles = await self.api.get_candles(symbol, '1h', 25)  # 25 candles to cover 24 hours
                if not candles or len(candles) < 24:
                    logger.debug(f"Not enough candle data for {symbol} performance calculation")
                    continue
                    
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = df.astype(float, errors='ignore')
                
                # Calculate 24h price change
                price_now = df['close'].iloc[-1]
                price_24h_ago = df['close'].iloc[-24]  # 24 hours ago
                price_change_24h = (price_now - price_24h_ago) / price_24h_ago
                
                # Calculate volatility
                volatility = (df['high'].max() - df['low'].min()) / df['low'].min()
                
                # Store the performance data
                coin_performance[symbol] = {
                    'price_change_24h': price_change_24h,
                    'volatility': volatility,
                    'current_price': price_now
                }
                
                if DEBUG_MODE:
                    logger.debug(f"{symbol} performance: 24h change={price_change_24h:.2%}, volatility={volatility:.2%}")
                    
            except Exception as e:
                logger.warning(f"Error calculating performance for {symbol}: {str(e)}")
                continue
        
        # Update the shared data
        shared_data['coin_performance'] = coin_performance
        
        # Update market conditions
        current_phase = _determine_market_phase(shared_data)
        global_trend = _determine_global_trend(shared_data)
        
        shared_data['market_conditions'] = {
            'current_phase': current_phase,
            'global_trend': global_trend
        }
        
        # Update trading statistics
        _update_trading_statistics(self)
        
        return coin_performance
    except Exception as e:
        logger.error(f"Error updating coin performance: {str(e)}")
        return {}

def _determine_market_phase(shared_data):
    """Determine the current market phase based on recent price action."""
    # Default to sideways if we can't determine
    try:
        # Use BTC as a reference for overall market phase
        if 'BTC/USDT:USDT' in shared_data.get('coin_performance', {}):
            btc_data = shared_data['coin_performance']['BTC/USDT:USDT']
            change = btc_data.get('price_change_24h', 0)
            
            if change > 0.03:  # 3% up
                return 'aufwärts'
            elif change < -0.03:  # 3% down
                return 'abwärts'
        
        return 'seitwärts'
    except Exception:
        return 'seitwärts'

def _determine_global_trend(shared_data):
    """Determine the global market trend."""
    try:
        # Count how many coins are up vs down
        up_coins = 0
        down_coins = 0
        
        for data in shared_data.get('coin_performance', {}).values():
            change = data.get('price_change_24h', 0)
            if change > 0.01:  # 1% up
                up_coins += 1
            elif change < -0.01:  # 1% down
                down_coins += 1
        
        if up_coins > down_coins * 2:  # Significantly more up coins
            return 'bullish'
        elif down_coins > up_coins * 2:  # Significantly more down coins
            return 'bearish'
        
        return 'neutral'
    except Exception:
        return 'neutral'

def _update_trading_statistics(self):
    """Update trading statistics for the dashboard."""
    try:
        from main import shared_data
        
        # Get completed trades from the last 7 days
        now = time.time()
        seven_days_ago = now - (7 * 24 * 60 * 60)
        
        recent_trades = [
            trade for trade in self.position_history 
            if trade.get('exit_time', 0) > seven_days_ago
        ]
        
        # Count trades per coin in the last 7 days
        trades_last_7d = {}
        for trade in recent_trades:
            symbol = trade.get('symbol', '')
            trades_last_7d[symbol] = trades_last_7d.get(symbol, 0) + 1
        
        # Calculate win rate
        if len(self.position_history) > 0:
            winning_trades = [t for t in self.position_history if t.get('pnl_percent', 0) > 0]
            win_rate = len(winning_trades) / len(self.position_history)
        else:
            win_rate = 0.0
        
        # Calculate average holding time
        if len(self.position_history) > 0:
            holding_times = [
                (t.get('exit_time', 0) - t.get('entry_time', 0)) / 3600  # Convert to hours
                for t in self.position_history if 'exit_time' in t and 'entry_time' in t
            ]
            avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
        else:
            avg_holding_time = 0
        
        # Update trading stats in shared data
        shared_data['trading_stats'] = {
            'trades_last_7d': trades_last_7d,
            'avg_holding_time': avg_holding_time,
            'win_rate': win_rate
        }
        
        # Update strategy stats
        _update_strategy_stats(self, shared_data)
        
    except Exception as e:
        logger.error(f"Error updating trading statistics: {str(e)}")

def _update_strategy_stats(self, shared_data):
    """Update strategy statistics for each coin."""
    try:
        strategy_stats = {}
        
        # Group trades by symbol
        trades_by_symbol = {}
        for trade in self.position_history:
            symbol = trade.get('symbol', '')
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Calculate stats for each symbol
        for symbol, trades in trades_by_symbol.items():
            if not trades:
                continue
            
            # Calculate win rate for this symbol
            winning_trades = [t for t in trades if t.get('pnl_percent', 0) > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # Calculate average holding time
            holding_times = [
                (t.get('exit_time', 0) - t.get('entry_time', 0)) / 3600  # Convert to hours
                for t in trades if 'exit_time' in t and 'entry_time' in t
            ]
            avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
            
            strategy_stats[symbol] = {
                'win_rate': win_rate,
                'avg_holding_time': avg_holding_time,
                'trade_count': len(trades)
            }
        
        # Update strategy stats in shared data
        shared_data['strategy_stats'] = strategy_stats
        
    except Exception as e:
        logger.error(f"Error updating strategy statistics: {str(e)}")