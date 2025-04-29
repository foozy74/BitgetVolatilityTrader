import os
import json
import time
import asyncio
import logging
import itertools
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

from backtesting import BacktestEngine
from bitget_api import BitgetAPI

# Setup logging
logger = logging.getLogger('trading_bot')

# File to store optimized strategies
STRATEGIES_FILE = 'optimized_strategies.json'

class StrategyOptimizer:
    def __init__(self):
        """Initialize the strategy optimizer."""
        self.api_key = os.environ.get('BITGET_API_KEY')
        self.api_secret = os.environ.get('BITGET_API_SECRET')
        self.api_passphrase = os.environ.get('BITGET_API_PASSPHRASE')
        
        # Enhanced parameter ranges for more comprehensive optimization
        self.param_ranges = {
            'rsi': {
                'period': range(7, 29, 3),          # 7, 10, 13, 16, 19, 22, 25, 28
                'overbought': range(65, 81, 3),     # 65, 68, 71, 74, 77, 80
                'oversold': range(20, 36, 3)        # 20, 23, 26, 29, 32, 35
            },
            'macd': {
                'fast': range(8, 17, 2),            # 8, 10, 12, 14, 16
                'slow': range(20, 35, 3),           # 20, 23, 26, 29, 32, 35
                'signal': range(7, 13, 1)           # 7, 8, 9, 10, 11, 12
            },
            'bb': {
                'period': range(15, 26, 2),         # 15, 17, 19, 21, 23, 25
                'deviation': [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]  # More granular standard deviations
            },
            'ema': {
                'fast': range(5, 22, 4),            # 5, 9, 13, 17, 21
                'slow': range(21, 56, 8)            # 21, 29, 37, 45, 53
            }
        }
        
        # Define market phases for more robust testing
        self.market_phases = {
            'uptrend': {'start_days_ago': 60, 'end_days_ago': 30},    # Historical uptrend period
            'downtrend': {'start_days_ago': 90, 'end_days_ago': 60},  # Historical downtrend period
            'sideways': {'start_days_ago': 30, 'end_days_ago': 0}     # Recent sideways/mixed period
        }
        
        # Load existing optimized strategies if available
        self.optimized_strategies = self._load_strategies()
    
    def _load_strategies(self):
        """Load optimized strategies from file."""
        try:
            if os.path.exists(STRATEGIES_FILE):
                with open(STRATEGIES_FILE, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading optimized strategies: {str(e)}")
            return {}
    
    def _save_strategies(self):
        """Save optimized strategies to file."""
        try:
            with open(STRATEGIES_FILE, 'w') as f:
                json.dump(self.optimized_strategies, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving optimized strategies: {str(e)}")
    
    async def fetch_historical_data(self, symbol, timeframe='1h', days=60):
        """Fetch historical price data for backtesting with extended history for better optimization."""
        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            logger.error("API credentials missing")
            return None
        
        try:
            async with BitgetAPI(self.api_key, self.api_secret, self.api_passphrase) as bitget:
                # Calculate number of candles needed based on timeframe
                candles_per_day = {
                    '15m': 96,   # 24 * 4
                    '1h': 24,
                    '4h': 6,
                    '1d': 1
                }
                
                limit = min(500, days * candles_per_day.get(timeframe, 24))
                candles = await bitget.get_candles(symbol, timeframe=timeframe, limit=limit)
                
                if not candles or len(candles) < 50:  # Require at least 50 candles
                    logger.warning(f"Insufficient historical data for {symbol}")
                    return None
                
                return candles
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def _generate_parameter_combinations(self, max_combinations=100):
        """Generate parameter combinations to test, limiting to a reasonable number."""
        # Create parameter combinations for each indicator
        rsi_combinations = list(itertools.product(
            self.param_ranges['rsi']['period'],
            self.param_ranges['rsi']['overbought'],
            self.param_ranges['rsi']['oversold']
        ))
        
        macd_combinations = list(itertools.product(
            self.param_ranges['macd']['fast'],
            self.param_ranges['macd']['slow'],
            self.param_ranges['macd']['signal']
        ))
        
        bb_combinations = list(itertools.product(
            self.param_ranges['bb']['period'],
            self.param_ranges['bb']['deviation']
        ))
        
        ema_combinations = list(itertools.product(
            self.param_ranges['ema']['fast'],
            self.param_ranges['ema']['slow']
        ))
        
        # Filter invalid combinations
        macd_combinations = [(fast, slow, signal) for fast, slow, signal in macd_combinations if fast < slow]
        ema_combinations = [(fast, slow) for fast, slow in ema_combinations if fast < slow]
        
        # If too many combinations, sample a subset
        if len(rsi_combinations) * len(macd_combinations) * len(bb_combinations) * len(ema_combinations) > max_combinations:
            # Take a more strategic approach instead of random sampling
            # Use a smaller subset of each parameter space
            rsi_combinations = rsi_combinations[::2]  # Take every other combination
            macd_combinations = macd_combinations[::2]
            bb_combinations = bb_combinations[::2]
            ema_combinations = ema_combinations[::2]
            
            # Still need to limit the total combinations
            total_combinations = len(rsi_combinations) * len(macd_combinations) * len(bb_combinations) * len(ema_combinations)
            if total_combinations > max_combinations:
                # Further reduce by taking alternating combinations
                rsi_combinations = rsi_combinations[::3]
                macd_combinations = macd_combinations[::3]
                
                # If still too many, take fixed smaller subsets
                if len(rsi_combinations) * len(macd_combinations) * len(bb_combinations) * len(ema_combinations) > max_combinations:
                    rsi_combinations = rsi_combinations[:3]
                    macd_combinations = macd_combinations[:3]
                    bb_combinations = bb_combinations[:3]
                    ema_combinations = ema_combinations[:3]
        
        # Create final parameter combinations
        all_combinations = []
        for rsi_params in rsi_combinations:
            for macd_params in macd_combinations:
                for bb_params in bb_combinations:
                    for ema_params in ema_combinations:
                        params = {
                            'rsi': {
                                'period': rsi_params[0],
                                'overbought': rsi_params[1],
                                'oversold': rsi_params[2]
                            },
                            'macd': {
                                'fast': macd_params[0],
                                'slow': macd_params[1],
                                'signal': macd_params[2]
                            },
                            'bb': {
                                'period': bb_params[0],
                                'deviation': bb_params[1]
                            },
                            'ema': {
                                'fast': ema_params[0],
                                'slow': ema_params[1]
                            }
                        }
                        all_combinations.append(params)
        
        logger.info(f"Generated {len(all_combinations)} parameter combinations for testing")
        return all_combinations[:max_combinations]
    
    async def optimize_strategy_for_coin(self, symbol, timeframe='1h', days=90, initial_capital=1000, max_combinations=100):
        """
        Optimize trading strategy for a specific coin across different market phases.
        
        Args:
            symbol (str): Trading symbol (e.g., "BTC/USDT:USDT")
            timeframe (str): Candle timeframe (e.g., "1h", "4h", "1d")
            days (int): Number of days of historical data to use (increased to 90 days by default)
            initial_capital (float): Initial capital for backtesting
            max_combinations (int): Maximum number of parameter combinations to test
            
        Returns:
            dict: Optimized strategy parameters
        """
        # Fetch comprehensive historical data (we need at least 90 days for market phase testing)
        logger.info(f"Optimizing strategy for {symbol} across multiple market phases...")
        all_candles = await self.fetch_historical_data(symbol, timeframe, max(days, 90))
        
        if all_candles is None or len(all_candles) < 50:
            logger.warning(f"Insufficient data for {symbol}, skipping optimization")
            return None
        
        # Generate parameter combinations to test
        parameter_combinations = self._generate_parameter_combinations(max_combinations)
        
        # Separate candles into market phases
        now = datetime.now()
        uptrend_start = now - timedelta(days=self.market_phases['uptrend']['start_days_ago'])
        uptrend_end = now - timedelta(days=self.market_phases['uptrend']['end_days_ago'])
        downtrend_start = now - timedelta(days=self.market_phases['downtrend']['start_days_ago'])
        downtrend_end = now - timedelta(days=self.market_phases['downtrend']['end_days_ago'])
        sideways_start = now - timedelta(days=self.market_phases['sideways']['start_days_ago'])
        
        # Parse candle timestamps and organize by market phase
        uptrend_candles = []
        downtrend_candles = []
        sideways_candles = []
        
        for candle in all_candles:
            timestamp = int(candle[0])
            candle_date = datetime.fromtimestamp(timestamp / 1000)
            
            if uptrend_start <= candle_date <= uptrend_end:
                uptrend_candles.append(candle)
            elif downtrend_start <= candle_date <= downtrend_end:
                downtrend_candles.append(candle)
            elif sideways_start <= candle_date:
                sideways_candles.append(candle)
        
        # Ensure we have enough data for each phase
        phase_data = {
            'uptrend': uptrend_candles if len(uptrend_candles) >= 50 else None,
            'downtrend': downtrend_candles if len(downtrend_candles) >= 50 else None,
            'sideways': sideways_candles if len(sideways_candles) >= 50 else None
        }
        
        valid_phases = [phase for phase, data in phase_data.items() if data is not None]
        if not valid_phases:
            logger.warning(f"Insufficient phase-specific data for {symbol}, falling back to standard optimization")
            phase_data = {'all': all_candles}
            valid_phases = ['all']
        
        logger.info(f"Testing on {len(valid_phases)} market phases for {symbol}: {', '.join(valid_phases)}")
        
        # Run backtests with different parameter combinations across market phases
        best_params = None
        best_total_score = -float('inf')
        best_phase_results = {}
        
        logger.info(f"Running {len(parameter_combinations)} backtests for {symbol}...")
        
        try:
            # Initialize progress tracking
            for i, params in enumerate(parameter_combinations):
                # Configure backtest engine with current parameters
                engine = BacktestEngine(
                    symbol=symbol,
                    timeframe=timeframe,
                    period_days=days,
                    initial_capital=initial_capital,
                    leverage=5,
                    position_size_pct=5.0,
                    stop_loss_pct=3.0,
                    take_profit_pct=10.0,
                    fee_pct=0.075
                )
                
                # Set strategy parameters
                strategy_params = {
                    'rsiEnabled': True,
                    'rsiPeriod': params['rsi']['period'],
                    'rsiOverbought': params['rsi']['overbought'],
                    'rsiOversold': params['rsi']['oversold'],
                    
                    'macdEnabled': True,
                    'macdFast': params['macd']['fast'],
                    'macdSlow': params['macd']['slow'],
                    'macdSignal': params['macd']['signal'],
                    
                    'bbEnabled': True,
                    'bbPeriod': params['bb']['period'],
                    'bbDeviation': params['bb']['deviation'],
                    
                    'emaEnabled': True,
                    'emaFast': params['ema']['fast'],
                    'emaSlow': params['ema']['slow'],
                    
                    'vwapEnabled': True,
                    'obvEnabled': True
                }
                
                engine.set_strategy_parameters(strategy_params)
                
                # Test strategy across different market phases
                phase_scores = {}
                total_score = 0
                valid_across_phases = True
                
                # Log progress every 10 combinations
                if i % 10 == 0:
                    logger.info(f"Tested {i}/{len(parameter_combinations)} combinations for {symbol}")
                
                # Test on each market phase
                for phase_name, phase_candles in phase_data.items():
                    # Skip if no data for this phase
                    if phase_candles is None or len(phase_candles) < 50:
                        continue
                        
                    # Load price data for this phase and run backtest
                    engine.load_price_data(phase_candles)
                    results = engine.run_backtest()
                    
                    # Extract key metrics
                    profit = results['metrics']['profit_loss']
                    trades_count = results['metrics']['total_trades']
                    win_rate = results['metrics']['win_rate']
                    max_drawdown = results['metrics']['max_drawdown']
                    sharpe_ratio = results['metrics']['sharpe_ratio']
                    
                    # Skip this parameter set if fewer than 5 trades in this phase
                    if trades_count < 5:
                        valid_across_phases = False
                        break
                    
                    # Skip if extreme drawdown in any phase
                    if max_drawdown > 0.35:
                        valid_across_phases = False
                        break
                        
                    # Skip if negative Sharpe ratio in any phase
                    if sharpe_ratio < 0:
                        valid_across_phases = False
                        break
                    
                    # Calculate trade frequency (trades per week)
                    phase_days = len(phase_candles) / 24 if timeframe == '1h' else len(phase_candles)  # Rough estimate
                    trades_per_week = (trades_count / phase_days) * 7
                    
                    # Skip extreme trade frequencies
                    if trades_per_week > 10 or trades_per_week < 0.5:
                        valid_across_phases = False
                        break
                        
                    # Calculate phase score with our balanced approach
                    phase_score = (profit * 0.4) + (win_rate * 100 * 0.3) + (sharpe_ratio * 0.2) - (max_drawdown * 100 * 0.1)
                    
                    # Store individual phase results
                    phase_scores[phase_name] = {
                        'score': phase_score,
                        'profit': profit,
                        'profit_percent': results['metrics']['total_return'] * 100,
                        'trades': trades_count,
                        'win_rate': win_rate * 100,
                        'max_drawdown': max_drawdown * 100,
                        'sharpe_ratio': sharpe_ratio
                    }
                    
                    # Add to total score, weighted by phase
                    # Give more weight to recent market conditions (sideways has higher weight)
                    phase_weights = {
                        'sideways': 0.5,    # Most recent/current market
                        'uptrend': 0.3,     # Historical uptrend
                        'downtrend': 0.2,   # Historical downtrend
                        'all': 1.0          # If we're using all data
                    }
                    
                    total_score += phase_score * phase_weights.get(phase_name, 1.0)
                
                # Skip if not valid across all tested phases
                if not valid_across_phases or not phase_scores:
                    continue
                    
                # Check if this is the best parameter set so far
                if total_score > best_total_score:
                    best_total_score = total_score
                    best_params = params
                    best_phase_results = phase_scores
        
        except Exception as e:
            logger.error(f"Error during backtest optimization for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Save optimized strategy
        if best_params is not None:
            # Extract the best results from phase results for reporting
            best_results = {}
            if best_phase_results:
                # Use 'all' phase if available, otherwise use sideways, or first available phase
                if 'all' in best_phase_results:
                    best_single_phase = best_phase_results['all']
                elif 'sideways' in best_phase_results:
                    best_single_phase = best_phase_results['sideways']
                else:
                    best_single_phase = list(best_phase_results.values())[0]
                
                best_results = {
                    'profit': best_single_phase['profit'],
                    'profit_percent': best_single_phase['profit_percent'],
                    'trades': best_single_phase['trades'],
                    'win_rate': best_single_phase['win_rate'],
                    'max_drawdown': best_single_phase['max_drawdown'],
                    'sharpe_ratio': best_single_phase['sharpe_ratio'],
                    'phase_scores': {phase: data['score'] for phase, data in best_phase_results.items()}
                }
            
            optimized_strategy = {
                'timestamp': int(datetime.now().timestamp()),
                'timeframe': timeframe,
                'parameters': best_params,
                'results': best_results,
                'market_phases_tested': list(best_phase_results.keys()) if best_phase_results else ['all']
            }
            
            self.optimized_strategies[symbol] = optimized_strategy
            self._save_strategies()
            
            logger.info(f"Strategy optimization completed for {symbol}")
            logger.info(f"Best parameters: {best_params}")
            
            if best_results:
                logger.info(f"Results: Profit: ${best_results['profit']:.2f} ({best_results['profit_percent']:.2f}%), "
                          f"Trades: {best_results['trades']}, Win Rate: {best_results['win_rate']:.2f}%, "
                          f"Max Drawdown: {best_results['max_drawdown']:.2f}%")
                if 'phase_scores' in best_results:
                    logger.info(f"Phase scores: {best_results['phase_scores']}")
            
            return best_params
        else:
            logger.warning(f"No viable strategy found for {symbol}")
            return None
    
    async def optimize_all_coins(self, coins, timeframe='1h', days=60, initial_capital=1000, max_combinations=100):
        """
        Optimize strategies for multiple coins with enhanced parameterization.
        
        Args:
            coins (list): List of coin symbols to optimize
            timeframe (str): Candle timeframe
            days (int): Number of days of historical data to use (increased to 60 for better results)
            initial_capital (float): Initial capital for backtesting
            max_combinations (int): Maximum number of parameter combinations to test per coin (increased to 100)
        """
        for symbol in coins:
            logger.info(f"Starting optimization for {symbol}")
            await self.optimize_strategy_for_coin(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                initial_capital=initial_capital,
                max_combinations=max_combinations
            )
            # Small delay between coins to avoid rate limiting
            await asyncio.sleep(1)
    
    def get_strategy_for_coin(self, symbol):
        """Get optimized strategy parameters for a specific coin."""
        strategy = self.optimized_strategies.get(symbol)
        if strategy:
            # Check if strategy is outdated (older than 3 days)
            strategy_time = datetime.fromtimestamp(strategy['timestamp'])
            if datetime.now() - strategy_time > timedelta(days=3):
                logger.warning(f"Strategy for {symbol} is outdated (last updated {strategy_time})")
            
            return strategy['parameters']
        return None
    
    async def retrain_strategies(self, coins, force=False):
        """
        Retrain all strategies, either on demand or if they're outdated.
        
        Args:
            coins (list): List of coin symbols to optimize
            force (bool): If True, retrain all strategies regardless of age
        """
        coins_to_optimize = []
        
        for symbol in coins:
            strategy = self.optimized_strategies.get(symbol)
            
            if strategy is None:
                # No existing strategy, add to optimization list
                coins_to_optimize.append(symbol)
            elif force:
                # Force retraining
                coins_to_optimize.append(symbol)
            else:
                # Check if strategy is outdated (older than 24 hours)
                strategy_time = datetime.fromtimestamp(strategy['timestamp'])
                if datetime.now() - strategy_time > timedelta(hours=24):
                    coins_to_optimize.append(symbol)
        
        if coins_to_optimize:
            logger.info(f"Retraining strategies for {len(coins_to_optimize)} coins")
            await self.optimize_all_coins(coins_to_optimize)
        else:
            logger.info("No strategies need retraining")


# Function to use in main.py
async def optimize_strategy_for_coin(symbol, timeframe='1h', days=30, max_combinations=50):
    """
    Optimize trading strategy for a specific coin.
    
    Args:
        symbol (str): Trading symbol (e.g., "BTC/USDT:USDT")
        timeframe (str): Candle timeframe (e.g., "1h", "4h", "1d")
        days (int): Number of days of historical data to use
        max_combinations (int): Maximum number of parameter combinations to test
        
    Returns:
        dict: Optimized strategy parameters
    """
    optimizer = StrategyOptimizer()
    return await optimizer.optimize_strategy_for_coin(symbol, timeframe, days, max_combinations=max_combinations)


# Function to retrain all strategies
async def retrain_strategies(coins, force=False):
    """
    Retrain all strategies for the given coins.
    
    Args:
        coins (list): List of coin symbols to optimize
        force (bool): If True, retrain all strategies regardless of age
    """
    optimizer = StrategyOptimizer()
    await optimizer.retrain_strategies(coins, force)


# Get optimized strategy for a coin
def get_strategy_for_coin(symbol):
    """
    Get optimized strategy parameters for a specific coin.
    
    Args:
        symbol (str): Trading symbol (e.g., "BTC/USDT:USDT")
        
    Returns:
        dict: Optimized strategy parameters or None if not available
    """
    optimizer = StrategyOptimizer()
    return optimizer.get_strategy_for_coin(symbol)


if __name__ == "__main__":
    # Example usage when run directly
    async def test():
        symbol = "BTC/USDT:USDT"
        optimizer = StrategyOptimizer()
        result = await optimizer.optimize_strategy_for_coin(symbol, max_combinations=20)
        print(f"Optimized strategy for {symbol}: {result}")
    
    # Run the test
    asyncio.run(test())