import numpy as np
import json
import logging
import os
import time
from datetime import datetime
import itertools
import random
from backtesting import BacktestEngine

# Setup logging
logger = logging.getLogger('trading_bot')

# Constants
OPTIMIZATION_HISTORY_FILE = 'optimization_history.json'

class BayesianOptimizer:
    """
    Performs Bayesian optimization for trading strategy parameters.
    This is a simplified implementation that approximates Bayesian optimization.
    """
    def __init__(self):
        self.history = self._load_history()
        self.current_iteration = 0
        self.max_iterations = 50
        self.exploration_factor = 0.3  # Controls exploration vs exploitation
        
        # Define parameter spaces - Expanded for better optimization
        self.param_ranges = {
            'rsi': {
                'period': list(range(7, 22, 1)),      # More granular: 7-21 by 1s
                'overbought': list(range(60, 86, 2)), # More granular: 60-84 by 2s
                'oversold': list(range(15, 41, 2))    # More granular: 15-39 by 2s
            },
            'macd': {
                'fast': list(range(6, 19, 1)),        # Expanded: 6-18 by 1s
                'slow': list(range(18, 36, 1)),       # More granular: 18-35 by 1s
                'signal': list(range(5, 15, 1))       # Expanded: 5-14 by 1s
            },
            'bb': {
                'period': list(range(10, 31, 2)),     # Expanded: 10-30 by 2s
                'deviation': [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0] # More options
            },
            'ema': {
                'fast': list(range(5, 18, 1)),        # More granular: 5-17 by 1s
                'slow': list(range(15, 36, 2))        # Expanded: 15-35 by 2s
            },
            'required_signals': list(range(2, 6, 1))  # 2, 3, 4, 5
        }
    
    def _load_history(self):
        """Load optimization history from file."""
        if os.path.exists(OPTIMIZATION_HISTORY_FILE):
            try:
                with open(OPTIMIZATION_HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading optimization history: {str(e)}")
        return {}
    
    def _save_history(self):
        """Save optimization history to file."""
        try:
            with open(OPTIMIZATION_HISTORY_FILE, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving optimization history: {str(e)}")
    
    def _params_to_key(self, params):
        """Convert parameters dict to a string key for the history."""
        key_parts = []
        for section in sorted(params.keys()):
            if isinstance(params[section], dict):
                for param, value in sorted(params[section].items()):
                    key_parts.append(f"{section}_{param}_{value}")
            else:
                key_parts.append(f"{section}_{params[section]}")
        return '|'.join(key_parts)
    
    def _get_random_params(self):
        """Generate random parameters from the parameter space."""
        params = {
            'rsi': {
                'period': random.choice(list(self.param_ranges['rsi']['period'])),
                'overbought': random.choice(list(self.param_ranges['rsi']['overbought'])),
                'oversold': random.choice(list(self.param_ranges['rsi']['oversold']))
            },
            'macd': {
                'fast': random.choice(list(self.param_ranges['macd']['fast'])),
                'slow': random.choice(list(self.param_ranges['macd']['slow'])),
                'signal': random.choice(list(self.param_ranges['macd']['signal']))
            },
            'bb': {
                'period': random.choice(list(self.param_ranges['bb']['period'])),
                'deviation': random.choice(self.param_ranges['bb']['deviation'])
            },
            'ema': {
                'fast': random.choice(list(self.param_ranges['ema']['fast'])),
                'slow': random.choice(list(self.param_ranges['ema']['slow']))
            },
            'required_signals': random.choice(list(self.param_ranges['required_signals']))
        }
        
        # Ensure MACD slow > fast
        while params['macd']['slow'] <= params['macd']['fast']:
            params['macd']['slow'] = random.choice(list(self.param_ranges['macd']['slow']))
        
        # Ensure EMA slow > fast
        while params['ema']['slow'] <= params['ema']['fast']:
            params['ema']['slow'] = random.choice(list(self.param_ranges['ema']['slow']))
        
        return params
    
    def _get_neighbors(self, params):
        """Generate parameter neighbors by modifying one parameter at a time."""
        neighbors = []
        
        # Helper function to modify a specific parameter
        def modify_param(section, param, current_value, possible_values):
            options = [v for v in possible_values if v != current_value]
            if options:
                new_params = json.loads(json.dumps(params))  # Deep copy
                new_params[section][param] = random.choice(options)
                return new_params
            return None
        
        # Generate neighbors for each parameter
        for section, section_params in params.items():
            if section == 'required_signals':
                options = [v for v in self.param_ranges[section] if v != section_params]
                if options:
                    new_params = json.loads(json.dumps(params))  # Deep copy
                    new_params[section] = random.choice(options)
                    neighbors.append(new_params)
            else:
                for param, value in section_params.items():
                    possible_values = list(self.param_ranges[section][param])
                    new_params = modify_param(section, param, value, possible_values)
                    if new_params:
                        neighbors.append(new_params)
        
        return neighbors
    
    def _evaluate_params(self, params, backtest_results):
        """
        Enhanced evaluation of parameter sets focusing on consistency, profitability, 
        and risk management, especially for highly volatile altcoins.
        Returns a score between 0 and 1, higher is better.
        """
        # Extract metrics from backtest results
        metrics = backtest_results['metrics']
        profit_pct = metrics['total_return'] * 100
        num_trades = metrics['total_trades']
        win_rate = metrics['win_rate'] * 100
        max_drawdown = metrics['max_drawdown'] * 100
        sharpe_ratio = metrics['sharpe_ratio']
        trades = backtest_results.get('trades', [])
        
        # Calculate trade frequency (trades per week)
        days = backtest_results.get('days', 30)  # Default to 30 if not provided
        trades_per_week = (num_trades / days) * 7
        
        # Skip if too few trades for reliable stats
        if num_trades < 8:  # Increased from 5 to ensure statistical significance
            return 0.1
            
        # Too many trades is also problematic (transaction costs, overtrading)
        if trades_per_week > 8:  # Reduced from 10 to avoid excessive trading
            return 0.3
            
        # Calculate additional metrics for advanced evaluation
        
        # 1. Calculate consecutive wins/losses
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        
        for trade in trades:
            profit_pct = trade.get('profit_percent', 0) * 100
            if profit_pct < 0:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_losses = 0
        
        # 2. Calculate profit/loss ratio
        win_amounts = [t.get('profit_percent', 0) for t in trades if t.get('profit_percent', 0) > 0]
        loss_amounts = [abs(t.get('profit_percent', 0)) for t in trades if t.get('profit_percent', 0) < 0]
        
        avg_win = sum(win_amounts) / len(win_amounts) if win_amounts else 0
        avg_loss = sum(loss_amounts) / len(loss_amounts) if loss_amounts else 1  # Avoid division by zero
        
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 3. Calculate recovery factor
        recovery_factor = abs(profit_pct / max_drawdown) if max_drawdown > 0 else 0
        
        # Calculate composite score (higher is better)
        score = 0.0
        
        # Profit contributes 30% to score (reduced from 40% to emphasize other factors)
        if profit_pct >= 15:  # Increased target for altcoin volatility
            profit_score = 1.0
        elif profit_pct >= 0:
            profit_score = profit_pct / 15
        else:
            profit_score = 0.0
        score += 0.30 * profit_score
        
        # Win rate contributes 25% to score (increased from 20%)
        if win_rate >= 55:  # Reward strategies with higher win rates
            win_rate_score = 1.0 + (win_rate - 55) / 45  # Bonus for excellent win rates
            win_rate_score = min(win_rate_score, 1.5)    # Cap bonus at 1.5
        elif win_rate >= 45:
            win_rate_score = (win_rate - 45) / 10
        else:
            win_rate_score = 0.0
        score += 0.25 * win_rate_score
        
        # Drawdown protection contributes 15% (same as before)
        if max_drawdown <= 5:  # Reward low drawdown
            drawdown_score = 1.0
        elif max_drawdown <= 15:
            drawdown_score = 1.0 - ((max_drawdown - 5) / 10)
        else:
            drawdown_score = 0.0
        score += 0.15 * drawdown_score
        
        # Avoid consecutive losses (10% weight - new metric)
        if max_consecutive_losses <= 3:
            consec_loss_score = 1.0
        elif max_consecutive_losses <= 6:
            consec_loss_score = 1.0 - ((max_consecutive_losses - 3) / 3)
        else:
            consec_loss_score = 0.0
        score += 0.10 * consec_loss_score
        
        # Profit/Loss ratio contributes 10% (new metric)
        if profit_loss_ratio >= 1.5:  # Good reward-to-risk ratio
            pl_ratio_score = 1.0
        elif profit_loss_ratio >= 1.0:
            pl_ratio_score = (profit_loss_ratio - 1.0) / 0.5
        else:
            pl_ratio_score = 0.0
        score += 0.10 * pl_ratio_score
        
        # Recovery factor contributes 5% (new metric)
        if recovery_factor >= 3.0:  # Excellent recovery capability
            recovery_score = 1.0
        elif recovery_factor >= 1.0:
            recovery_score = recovery_factor / 3.0
        else:
            recovery_score = 0.0
        score += 0.05 * recovery_score
        
        # Sharpe ratio contributes 5% (reduced from 15%)
        if sharpe_ratio >= 2:  # Lowered threshold for volatile markets
            sharpe_score = 1.0
        elif sharpe_ratio >= -1:
            sharpe_score = (sharpe_ratio + 1) / 3
        else:
            sharpe_score = 0.0
        score += 0.05 * sharpe_score
        
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, score))
    
    def _select_next_params(self, symbol):
        """
        Select the next parameters to evaluate using 
        a combination of exploitation and exploration.
        """
        self.current_iteration += 1
        symbol_history = self.history.get(symbol, {})
        
        # If we have few evaluations, generate random parameters
        if len(symbol_history) < 5 or random.random() < self.exploration_factor:
            return self._get_random_params()
        
        # Find the best params so far
        best_params_key = max(symbol_history, key=lambda k: symbol_history[k]['score'])
        best_params = symbol_history[best_params_key]['params']
        
        # Generate neighbors of the best params
        neighbors = self._get_neighbors(best_params)
        
        # Filter out neighbors we've already evaluated
        unseen_neighbors = []
        for params in neighbors:
            params_key = self._params_to_key(params)
            if params_key not in symbol_history:
                unseen_neighbors.append(params)
        
        # If we have unseen neighbors, pick one randomly
        if unseen_neighbors:
            return random.choice(unseen_neighbors)
        
        # Otherwise, generate new random parameters
        return self._get_random_params()
    
    async def optimize_parameters(self, symbol, timeframe, backtest_engine, historical_data, initial_capital=1000):
        """
        Run Bayesian optimization for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            backtest_engine: Initialized BacktestEngine
            historical_data: Historical price data
            initial_capital: Initial capital for backtesting
            
        Returns:
            dict: Optimized parameters
        """
        logger.info(f"Starting Bayesian optimization for {symbol}")
        
        if symbol not in self.history:
            self.history[symbol] = {}
        
        best_score = 0.0
        best_params = None
        best_results = None
        
        for i in range(self.max_iterations):
            logger.info(f"Optimization iteration {i+1}/{self.max_iterations} for {symbol}")
            
            # Select next parameters to evaluate
            params = self._select_next_params(symbol)
            params_key = self._params_to_key(params)
            
            # Skip if we've already evaluated these exact parameters
            if params_key in self.history.get(symbol, {}):
                logger.info(f"Skipping already evaluated parameters")
                continue
            
            try:
                # Configure backtest engine with parameters
                backtest_engine.set_strategy_parameters({
                    'rsi_period': params['rsi']['period'],
                    'rsi_overbought': params['rsi']['overbought'],
                    'rsi_oversold': params['rsi']['oversold'],
                    'macd_fast': params['macd']['fast'],
                    'macd_slow': params['macd']['slow'],
                    'macd_signal': params['macd']['signal'],
                    'bb_period': params['bb']['period'],
                    'bb_stddev': params['bb']['deviation'],
                    'ema_fast': params['ema']['fast'],
                    'ema_slow': params['ema']['slow'],
                    'required_signals': params['required_signals']
                })
                
                # Load data and run backtest
                backtest_engine.load_price_data(historical_data)
                results = backtest_engine.run_backtest()
                
                # Evaluate parameters
                score = self._evaluate_params(params, results)
                
                # Save evaluation results
                self.history[symbol][params_key] = {
                    'params': params,
                    'score': score,
                    'profit': results['metrics']['profit_loss'],
                    'profit_percent': results['metrics']['total_return'] * 100,
                    'trades': results['metrics']['total_trades'],
                    'win_rate': results['metrics']['win_rate'] * 100,
                    'max_drawdown': results['metrics']['max_drawdown'] * 100,
                    'sharpe_ratio': results['metrics']['sharpe_ratio'],
                    'timestamp': datetime.now().timestamp()
                }
                
                # Update best parameters if we found a better score
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_results = results
                    logger.info(f"New best parameters found: score={score:.4f}, profit={results['metrics']['profit_loss']:.2f}, "
                               f"win_rate={results['metrics']['win_rate']*100:.2f}%")
                
                # Save history after each evaluation
                self._save_history()
            
            except Exception as e:
                logger.error(f"Error during parameter evaluation: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        if best_params is None:
            logger.warning(f"No viable parameters found for {symbol}")
            return None
        
        logger.info(f"Optimization completed for {symbol}")
        logger.info(f"Best parameters: {json.dumps(best_params, indent=2)}")
        logger.info(f"Best score: {best_score:.4f}")
        if best_results:
            metrics = best_results['metrics']
            logger.info(f"Results: Profit: ${metrics['profit_loss']:.2f} ({metrics['total_return']*100:.2f}%), "
                       f"Trades: {metrics['total_trades']}, Win Rate: {metrics['win_rate']*100:.2f}%, "
                       f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        return best_params

# Global instance for use in main.py
bayesian_optimizer = BayesianOptimizer()