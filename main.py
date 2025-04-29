import os
import time
import logging
import asyncio
import traceback
import datetime
from dotenv import load_dotenv
import pandas as pd
import json

from bitget_api import BitgetAPI
from indicators import analyze_coin, should_enter_trade, calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_ema
from telegram_bot import send_telegram_message
from utils import setup_logger, keep_alive
from app import app  # Import Flask app
from strategy_optimizer import get_strategy_for_coin, optimize_strategy_for_coin, retrain_strategies
from adaptive_strategy import adaptive_strategy  # Import our adaptive strategy instance
from bayesian_optimizer import bayesian_optimizer  # Import our Bayesian optimizer instance
from extra_functions import update_coin_performance

# Lade existierende optimierte Strategien aus der JSON-Datei
STRATEGIES_FILE = 'optimized_strategies.json'
optimized_strategies = {}
try:
    if os.path.exists(STRATEGIES_FILE):
        with open(STRATEGIES_FILE, 'r') as f:
            optimized_strategies = json.load(f)
except Exception as e:
    print(f"Error loading optimized strategies: {str(e)}")

# Geteilte Daten mit dem Dashboard
shared_data = {
    'top_volatile_coins': [],
    'active_trades': {},
    'last_analysis_time': time.time(),
    'last_analysis_result': 'Starting bot...',
    'new_activities': [],
    'completed_trades': [],
    'portfolio_balance': None,
    'portfolio_drawdown': 0.0,
    'risk_level': 'normal',
    'optimized_strategies': optimized_strategies,  # Lade existierende Strategien aus der Datei
    # Bot Einstellungen
    'bot_settings': {
        'max_positions': 8            # Maximale Anzahl gleichzeitiger Positionen
    },
    # Neue Daten für erweitertes Dashboard
    'coin_performance': {},           # 24h Preisveränderung und Volatilität pro Coin
    'strategy_stats': {},             # Win-Rate und durchschnittliche Haltedauer pro Strategie
    'trading_stats': {                # Allgemeine Handelsstatistiken
        'trades_last_7d': {},         # Anzahl der Trades pro Coin in den letzten 7 Tagen
        'avg_holding_time': 0,        # Durchschnittliche Haltedauer aller Positionen in Stunden
        'win_rate': 0.0               # Gesamt-Win-Rate (0-1)
    },
    'market_conditions': {            # Marktbedingungen
        'current_phase': 'seitwärts', # Aktuelle Marktphase (aufwärts, abwärts, seitwärts)
        'global_trend': 'neutral'     # Globaler Markttrend (bullish, bearish, neutral)
    }
}

# Set to True to enable detailed debug logs
DEBUG_MODE = True

# Load environment variables
load_dotenv()

# Override placeholder environment variables with secrets for API keys if they exist
if os.environ.get('BITGET_API_KEY') == 'your_api_key_here' and os.environ.get('BITGET_API_SECRET') == 'your_api_secret_here' and os.environ.get('BITGET_API_PASSPHRASE') == 'your_api_passphrase_here':
    # Ensure we use the secret values, not the placeholders in .env
    logger = logging.getLogger('trading_bot')
    logger.info("Using API keys from Replit Secrets instead of .env placeholders")

# Set up logging
logger = setup_logger()

class TradingBot:
    def __init__(self):
        """Initialize the trading bot with configurations and API connections."""
        # Load configuration from environment variables
        self.api_key = os.getenv('BITGET_API_KEY')
        self.api_secret = os.getenv('BITGET_API_SECRET')
        self.api_passphrase = os.getenv('BITGET_API_PASSPHRASE')
        
        # Trading parameters - Optimized for more aggressive trading
        self.leverage = 5
        self.stop_loss_percent = 0.035  # 3.5% (increased from 3%)
        self.take_profit_percent = 0.10  # 10%
        self.volatility_period = '1h'  # or '24h'
        self.analysis_interval = 300  # 5 minutes (reduced from 10 minutes) for more trading opportunities
        
        # Risk management parameters
        self.max_drawdown_percent = 0.18  # 18% max drawdown allowed before risk reduction (increased from 15%)
        self.critical_drawdown_percent = 0.27  # 27% max drawdown before stopping all trading (increased from 25%)
        self.position_size_percent = 0.07  # 7% of available balance per trade (increased from 5%)
        
        # Portfolio tracking
        self.initial_balance = None  # Will be set on first account check
        self.highest_balance = None  # Track highest balance for drawdown calculation
        self.current_balance = None  # Current account balance
        self.current_drawdown = 0.0  # Current drawdown as percentage
        self.risk_level = "normal"   # normal, reduced, or suspended
        
        # Active positions tracking
        self.active_positions = {}  # symbol -> position_data
        # Verwenden der Einstellung aus shared_data, falls vorhanden
        self.max_positions = shared_data['bot_settings'].get('max_positions', 8)  # Max. gleichzeitige Positionen
        self.position_history = []  # Track closed positions for performance analysis
        
        # Strategy optimization
        self.use_optimized_strategies = True  # Set to True to use optimized strategies
        self.optimize_interval = 24  # Hours between strategy optimizations
        self.last_optimization_time = None  # Will be set when strategies are optimized
        
        # Initialize API connection
        self.api = BitgetAPI(self.api_key, self.api_secret, self.api_passphrase)
        
        logger.info("Trading bot initialized with max drawdown protection and strategy optimization")
        
    # Methode zur Initialisierung der Performance-Daten
    async def initialize_performance_data(self):
        """
        Initialisiert die Performance-Daten für das Dashboard bei Programmstart.
        """
        logger.info("Initializing performance data for dashboard...")
        try:
            # Wichtige Coins, die immer analysiert werden sollen
            important_coins = ["ETH/USDT:USDT", "SOL/USDT:USDT", "ADA/USDT:USDT", "XRP/USDT:USDT", "BTC/USDT:USDT"]
            
            # Füge Bitcoin explizit hinzu für Marktanalyse
            if "BTC/USDT:USDT" not in important_coins:
                important_coins.append("BTC/USDT:USDT")
                
            # Update performance data
            await self.update_coin_performance(important_coins)
            
            logger.info(f"Performance data initialized for {len(important_coins)} coins")
            
            # Protokolliere die Initialisierung
            shared_data['new_activities'].append({
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': f"Dashboard performance data initialized",
                'type': 'info'
            })
        except Exception as e:
            logger.error(f"Error initializing performance data: {str(e)}")
            if DEBUG_MODE:
                logger.error(f"Detailed error traceback: {traceback.format_exc()}")
    
    # Add the update_coin_performance method directly to the TradingBot class
    async def update_coin_performance(self, coins_to_analyze):
        """
        Update coin performance data for the dashboard.
        
        Args:
            coins_to_analyze (list): List of coins to gather performance data
        """
        try:
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
            market_phase = 'seitwärts'  # Default to sideways
            global_trend = 'neutral'    # Default to neutral
            
            # Use BTC as a reference for overall market phase
            if 'BTC/USDT:USDT' in coin_performance:
                btc_data = coin_performance['BTC/USDT:USDT']
                change = btc_data.get('price_change_24h', 0)
                
                if change > 0.03:  # 3% up
                    market_phase = 'aufwärts'
                elif change < -0.03:  # 3% down
                    market_phase = 'abwärts'
            
            # Count how many coins are up vs down
            up_coins = 0
            down_coins = 0
            for data in coin_performance.values():
                change = data.get('price_change_24h', 0)
                if change > 0.01:  # 1% up
                    up_coins += 1
                elif change < -0.01:  # 1% down
                    down_coins += 1
            
            if up_coins > down_coins * 2:  # Significantly more up coins
                global_trend = 'bullish'
            elif down_coins > up_coins * 2:  # Significantly more down coins
                global_trend = 'bearish'
            
            shared_data['market_conditions'] = {
                'current_phase': market_phase,
                'global_trend': global_trend
            }
            
            # Update trading statistics if we have trade history
            if hasattr(self, 'position_history') and self.position_history:
                # Get completed trades from the last 7 days
                now = time.time()
                seven_days_ago = now - (7 * 24 * 60 * 60)
                
                recent_trades = [
                    trade for trade in self.position_history 
                    if trade.get('close_time', 0) > seven_days_ago
                ]
                
                # Count trades per coin in the last 7 days
                trades_last_7d = {}
                for trade in recent_trades:
                    symbol = trade.get('symbol', '')
                    trades_last_7d[symbol] = trades_last_7d.get(symbol, 0) + 1
                
                # Calculate win rate
                winning_trades = [t for t in self.position_history if t.get('pnl_percent', 0) > 0]
                win_rate = len(winning_trades) / len(self.position_history) if self.position_history else 0.0
                
                # Calculate average holding time
                # In unserer Implementierung verwenden wir 'close_time' für den Schließungszeitpunkt
                # und 'open_time' für den Eröffnungszeitpunkt. Prüfen auf beide Felder.
                holding_times = []
                for trade in self.position_history:
                    close_time = trade.get('close_time', 0)
                    if close_time == 0:  # Alternativer Feldname könnte verwendet worden sein
                        close_time = trade.get('exit_time', 0)
                        
                    open_time = trade.get('open_time', 0)
                    if open_time == 0:  # Alternativer Feldname könnte verwendet worden sein
                        open_time = trade.get('entry_time', 0)
                        
                    if close_time > 0 and open_time > 0:
                        holding_times.append((close_time - open_time) / 3600)  # In Stunden umrechnen
                avg_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
                
                # Update trading stats in shared data
                shared_data['trading_stats'] = {
                    'trades_last_7d': trades_last_7d,
                    'avg_holding_time': avg_holding_time,
                    'win_rate': win_rate
                }
            
            return coin_performance
        except Exception as e:
            logger.error(f"Error updating coin performance: {str(e)}")
            return {}
    
    async def find_volatile_coins(self):
        """
        Analyze market and find top 5 most volatile altcoins.
        """
        logger.info("Analyzing market volatility...")
        
        try:
            # Get all tradable futures symbols
            markets = await self.api.get_futures_markets()
            
            if DEBUG_MODE:
                logger.info(f"Retrieved {len(markets)} markets from Bitget API")
                # Log a sample of markets to help with debugging symbol formats
                sample_markets = markets[:10] if len(markets) > 10 else markets
                logger.info(f"Sample markets format: {sample_markets}")
            
            # Calculate volatility for each coin
            volatilities = []
            
            # Track symbols that have enough data for analysis
            processable_symbols = set()
            
            # Process all markets
            for symbol in markets:
                # Skip Bitcoin, only look at altcoins
                if symbol == 'BTC/USDT:USDT':
                    if DEBUG_MODE:
                        logger.debug(f"Skipping {symbol} (BTC)")
                    continue
                    
                # Get candle data
                candles = await self.api.get_candles(symbol, self.volatility_period)
                
                if not candles or len(candles) < 2:
                    if DEBUG_MODE:
                        logger.debug(f"Skipping {symbol}: Not enough candle data (got {len(candles) if candles else 0} candles)")
                    continue
                
                # Mark this symbol as having enough data
                processable_symbols.add(symbol)
                
                # Calculate volatility (high - low) / low
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = df.astype(float, errors='ignore')
                
                # Calculate volatility as (high-low)/low
                volatility = (df['high'].max() - df['low'].min()) / df['low'].min()
                
                if DEBUG_MODE:
                    logger.debug(f"{symbol} volatility: {volatility:.4f}, high: {df['high'].max():.4f}, low: {df['low'].min():.4f}")
                
                # Add to general volatility list
                volatilities.append({
                    'symbol': symbol,
                    'volatility': volatility
                })
            
            # Sort by volatility and get top 5
            volatilities.sort(key=lambda x: x['volatility'], reverse=True)
            top_volatile_coins = volatilities[:5]
            
            # Log the final list
            if DEBUG_MODE:
                # Log top volatile coins
                volatility_details = [(coin['symbol'], f"{coin['volatility']:.4f}") for coin in top_volatile_coins]
                logger.info(f"Top volatile coins with values: {volatility_details}")
                
                # Log final list of coins to analyze
                all_coins = [coin['symbol'] for coin in top_volatile_coins]
                logger.info(f"Final list of coins to analyze (top volatile only): {all_coins}")
            else:
                logger.info(f"Analyzing {len(top_volatile_coins)} coins: Top 5 volatile coins")
            
            # Update shared data for dashboard
            coin_symbols = [coin['symbol'] for coin in top_volatile_coins]
            shared_data['top_volatile_coins'] = coin_symbols
            
            # Debug für Dashboard-Daten
            logger.info(f"DASHBOARD DATA UPDATE: top_volatile_coins = {coin_symbols}")
            shared_data['last_analysis_time'] = time.time()
            shared_data['last_analysis_result'] = f"Analyzed {len(markets)} coins, tracking {len(top_volatile_coins)} (top volatile only)"
            
            # Speichere die volatilen Coins auch in einer eigenen Datei für den Zugriff durch die Flask-App
            try:
                with open('volatile_coins.json', 'w') as f:
                    json.dump(coin_symbols, f)
                logger.info(f"Saved volatile coins to volatile_coins.json: {coin_symbols}")
            except Exception as e:
                logger.error(f"Error saving volatile coins to file: {e}")
                
            # Debug-Ausgabe für Top-Volatile-Coins
            logger.info(f"DASHBOARD DATA UPDATE: Setting top_volatile_coins = {coin_symbols}")
            
            shared_data['new_activities'].append({
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': f"Analyzing market volatility for top {len(top_volatile_coins)} coins: {', '.join([c.split('/')[0] for c in coin_symbols])}",
                'type': 'info'
            })
                
            return top_volatile_coins
            
        except Exception as e:
            logger.error(f"Error finding volatile coins: {str(e)}")
            if DEBUG_MODE:
                logger.error(f"Detailed traceback: {traceback.format_exc()}")
            
            # Update shared data for dashboard
            shared_data['last_analysis_time'] = time.time()
            shared_data['last_analysis_result'] = f"Error analyzing market: {str(e)}"
            shared_data['new_activities'].append({
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': f"Error finding volatile coins: {str(e)}",
                'type': 'danger'
            })
            
            return []
    
    async def check_portfolio_status(self):
        """
        Check portfolio status and calculate current drawdown.
        Returns:
            bool: True if trading is allowed, False if trading should be suspended
        """
        try:
            # Get current account balance
            account = await self.api.get_account_info()
            current_balance = float(account['total'])
            self.current_balance = current_balance
            
            # Initialize initial and highest balance values if not set
            if self.initial_balance is None:
                self.initial_balance = current_balance
                self.highest_balance = current_balance
                logger.info(f"Initial account balance: {self.initial_balance} USDT")
                
                # Update shared data with initial balance
                shared_data['portfolio_balance'] = current_balance
                shared_data['portfolio_drawdown'] = 0.0
                shared_data['risk_level'] = self.risk_level
                
                # Add to activities log
                shared_data['new_activities'].append({
                    'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"Initial portfolio balance: {current_balance:.2f} USDT",
                    'type': 'info'
                })
                
                return True
                
            # Update highest balance if current balance is higher
            if self.highest_balance is None or current_balance > self.highest_balance:
                self.highest_balance = current_balance
            
            # Calculate drawdown
            if self.highest_balance is not None and self.highest_balance > 0:
                self.current_drawdown = (self.highest_balance - current_balance) / self.highest_balance
                
                # Update shared data for dashboard
                shared_data['portfolio_balance'] = current_balance
                shared_data['portfolio_drawdown'] = self.current_drawdown
                shared_data['risk_level'] = self.risk_level
                
                # Log drawdown status
                if self.current_drawdown > 0:
                    logger.info(f"Current drawdown: {self.current_drawdown:.2%} (Balance: {current_balance:.2f} USDT, Highest: {self.highest_balance:.2f} USDT)")
                    
                    # Add to shared data for dashboard
                    shared_data['new_activities'].append({
                        'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'message': f"Portfolio drawdown: {self.current_drawdown:.2%}",
                        'type': 'warning' if self.current_drawdown > self.max_drawdown_percent else 'info'
                    })
                
                # Update risk level based on drawdown
                if self.current_drawdown >= self.critical_drawdown_percent:
                    if self.risk_level != "suspended":
                        logger.warning(f"CRITICAL DRAWDOWN REACHED: {self.current_drawdown:.2%} - All trading suspended!")
                        self.risk_level = "suspended"
                        
                        # Send emergency alert via Telegram
                        await send_telegram_message(f"⚠️ EMERGENCY: Critical drawdown of {self.current_drawdown:.2%} reached. Trading suspended.")
                        
                        # Close all positions to prevent further losses
                        await self.close_all_positions("critical drawdown protection")
                    return False
                    
                elif self.current_drawdown >= self.max_drawdown_percent:
                    if self.risk_level != "reduced":
                        logger.warning(f"Max drawdown threshold reached: {self.current_drawdown:.2%} - Reducing position size")
                        self.risk_level = "reduced"
                        self.position_size_percent = 0.02  # Reduce position size to 2% of balance
                        
                        # Alert via Telegram
                        await send_telegram_message(f"⚠️ WARNING: Drawdown of {self.current_drawdown:.2%} reached. Reducing position sizes.")
                    return True
                    
                else:
                    if self.risk_level != "normal":
                        logger.info(f"Drawdown below threshold: {self.current_drawdown:.2%} - Resuming normal trading")
                        self.risk_level = "normal"
                        self.position_size_percent = 0.05  # Normal position size at 5% of balance
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking portfolio status: {str(e)}")
            if DEBUG_MODE:
                logger.error(traceback.format_exc())
            return True  # Default to allowing trading if there's an error
    
    async def close_all_positions(self, reason):
        """Close all active positions."""
        logger.warning(f"Closing all positions due to: {reason}")
        
        positions_to_close = list(self.active_positions.keys())
        for symbol in positions_to_close:
            await self.close_position(symbol, reason)
            
        return len(positions_to_close)
    
    async def check_and_optimize_strategies(self, coin_list, force_optimize=False):
        """
        Check if strategies need to be optimized, and optimize them if needed.
        
        Args:
            coin_list (list): List of coins to optimize strategies for
            force_optimize (bool): Force optimization regardless of time interval
        """
        if not self.use_optimized_strategies:
            return
            
        current_time = datetime.datetime.now()
        
        # Initialize last optimization time if needed
        if not force_optimize:
            if self.last_optimization_time is None:
                logger.info("Initial strategy optimization needed")
                force_optimize = True
            else:
                # Check if we need to optimize based on time interval
                time_since_optimization = current_time - self.last_optimization_time
                hours_since_optimization = time_since_optimization.total_seconds() / 3600
                
                force_optimize = hours_since_optimization >= self.optimize_interval
                
                if force_optimize:
                    logger.info(f"Strategies need retraining (last update: {self.last_optimization_time})")
        else:
            logger.info("Forcing strategy optimization for all coins")
        
        if force_optimize:
            try:
                # Get symbols from coin list
                symbols = []
                for coin_data in coin_list:
                    if isinstance(coin_data, dict) and 'symbol' in coin_data:
                        symbols.append(coin_data['symbol'])
                    else:
                        symbols.append(coin_data)
                
                logger.info(f"Optimizing strategies for {len(symbols)} coins")
                
                # Add to activities log
                shared_data['new_activities'].append({
                    'time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"Starting strategy optimization for {len(symbols)} coins",
                    'type': 'info'
                })
                
                # Run optimization for all coins
                await retrain_strategies(symbols, force=True)
                
                # Update optimization timestamp
                self.last_optimization_time = current_time
                
                # Add to activities log
                shared_data['new_activities'].append({
                    'time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"Strategy optimization completed",
                    'type': 'success'
                })
                
                logger.info("Strategy optimization completed")
                
            except Exception as e:
                logger.error(f"Error optimizing strategies: {str(e)}")
                if DEBUG_MODE:
                    logger.error(traceback.format_exc())
                
                # Add to activities log
                shared_data['new_activities'].append({
                    'time': current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"Strategy optimization failed: {str(e)}",
                    'type': 'danger'
                })
    
    async def analyze_market(self):
        """Main market analysis function to find trading opportunities."""
        # First check portfolio status to implement drawdown protection
        trading_allowed = await self.check_portfolio_status()
        
        if not trading_allowed:
            logger.warning("Trading suspended due to risk management rules (max drawdown exceeded)")
            shared_data['last_analysis_result'] = "Trading suspended due to risk management rules"
            return
            
        # If in reduced risk mode, add to shared data
        if self.risk_level == "reduced":
            shared_data['last_analysis_result'] = f"Trading with reduced position size (Drawdown: {self.current_drawdown:.2%})"
        
        top_volatile_coins = await self.find_volatile_coins()
        
        # Update coin performance data for dashboard
        await update_coin_performance(self, top_volatile_coins)
        
        # Check and optimize strategies for all volatile coins after finding them
        # Force optimize every time we find new volatile coins
        logger.info(f"Starting optimization for top {len(top_volatile_coins)} volatile coins")
        await self.check_and_optimize_strategies(top_volatile_coins, force_optimize=True)
        
        for coin in top_volatile_coins:
            symbol = coin['symbol']
            
            # Skip if already have an active position for this coin
            if symbol in self.active_positions:
                logger.info(f"Skipping {symbol} analysis as position is already open")
                continue
            
            # Get detailed candle data for analysis
            candles = await self.api.get_candles(symbol, '1h', limit=100)
            
            if not candles or len(candles) < 50:  # Need enough data for indicators
                logger.warning(f"Not enough data for {symbol}, skipping")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype(float, errors='ignore')
            
            if DEBUG_MODE:
                logger.debug(f"Analyzing {symbol} with {len(df)} candles")
                logger.debug(f"Last candle: Open={df['open'].iloc[-1]:.4f}, Close={df['close'].iloc[-1]:.4f}, High={df['high'].iloc[-1]:.4f}, Low={df['low'].iloc[-1]:.4f}")
            
            # Get optimized strategy if available
            optimized_strategy = None
            if self.use_optimized_strategies:
                optimized_strategy = get_strategy_for_coin(symbol)
                if optimized_strategy:
                    if DEBUG_MODE:
                        logger.debug(f"Using optimized strategy for {symbol}: {json.dumps(optimized_strategy, indent=2)}")
                        
                    # Update indicators based on optimized parameters
                    # This will recalculate the indicators with the optimal parameters
                    if 'rsi' in optimized_strategy:
                        rsi_params = optimized_strategy['rsi']
                        df = calculate_rsi(df, period=rsi_params['period'])
                        
                    if 'macd' in optimized_strategy:
                        macd_params = optimized_strategy['macd']
                        df = calculate_macd(df, 
                                           fast=macd_params['fast'], 
                                           slow=macd_params['slow'], 
                                           signal=macd_params['signal'])
                        
                    if 'bb' in optimized_strategy:
                        bb_params = optimized_strategy['bb']
                        df = calculate_bollinger_bands(df, 
                                                     period=bb_params['period'], 
                                                     std_dev=bb_params['deviation'])
                    
                    if 'ema' in optimized_strategy:
                        ema_params = optimized_strategy['ema']
                        df = calculate_ema(df, period=ema_params['fast'])
                        df = calculate_ema(df, period=ema_params['slow'])
                    
                    # Store in shared data for dashboard
                    if symbol not in shared_data['optimized_strategies']:
                        shared_data['optimized_strategies'][symbol] = optimized_strategy
            
            # Analyze indicators
            analysis_results = analyze_coin(df)
            
            # Log detailed analysis results in debug mode
            if DEBUG_MODE:
                analysis_summary = {
                    'rsi': {
                        'value': f"{analysis_results['rsi']['value']:.2f}",
                        'signal': analysis_results['rsi']['signal']
                    },
                    'macd': {
                        'value': f"{analysis_results['macd']['value']:.6f}",
                        'signal_value': f"{analysis_results['macd']['signal_value']:.6f}",
                        'signal': analysis_results['macd']['signal']
                    },
                    'vwap': {
                        'value': f"{analysis_results['vwap']['value']:.4f}",
                        'close': f"{analysis_results['close']:.4f}",
                        'signal': analysis_results['vwap']['signal']
                    },
                    'bb': {
                        'upper': f"{analysis_results['bollinger_bands']['upper']:.4f}",
                        'lower': f"{analysis_results['bollinger_bands']['lower']:.4f}",
                        'signal': analysis_results['bollinger_bands']['signal']
                    },
                    'sma': {
                        'sma20': f"{analysis_results['sma']['sma20']:.4f}", 
                        'sma50': f"{analysis_results['sma']['sma50']:.4f}",
                        'signal': analysis_results['sma']['signal']
                    }
                }
                
                # Add optimization info to log
                if optimized_strategy:
                    analysis_summary['using_optimized_strategy'] = {"status": "yes"}
                
                logger.debug(f"{symbol} analysis: {json.dumps(analysis_summary, indent=2)}")
            
            # Use adaptive strategy instead of fixed signal threshold
            # Traditional approach - use fixed required signals count
            traditional_should_trade, traditional_direction = should_enter_trade(analysis_results)
            
            # Adaptive approach - dynamically adjust signal requirements
            adaptive_should_trade, adaptive_direction = adaptive_strategy.should_enter_trade(df, symbol, analysis_results)
            
            # Log comparison between traditional and adaptive approaches
            if DEBUG_MODE:
                bullish_count = sum(1 for k, v in analysis_results.items() 
                                 if isinstance(v, dict) and 'signal' in v and v['signal'] == 'buy')
                bearish_count = sum(1 for k, v in analysis_results.items() 
                                  if isinstance(v, dict) and 'signal' in v and v['signal'] == 'sell')
                logger.debug(f"{symbol} signal strength: {bullish_count} bullish, {bearish_count} bearish")
                logger.debug(f"{symbol} traditional strategy: {traditional_should_trade}, {traditional_direction}")
                logger.debug(f"{symbol} adaptive strategy: {adaptive_should_trade}, {adaptive_direction}")
            
            # Use the adaptive strategy for trade decisions
            should_trade = adaptive_should_trade
            direction = adaptive_direction
            
            if should_trade:
                logger.info(f"Trade opportunity found for {symbol}, direction: {direction}")
                # Pass analysis results to record for learning from trade outcomes
                await self.execute_trade(symbol, direction, analysis_results)
            else:
                if DEBUG_MODE:
                    bullish_count = sum(1 for k, v in analysis_results.items() 
                                     if isinstance(v, dict) and 'signal' in v and v['signal'] == 'buy')
                    bearish_count = sum(1 for k, v in analysis_results.items() 
                                      if isinstance(v, dict) and 'signal' in v and v['signal'] == 'sell')
                    neutral_count = sum(1 for k, v in analysis_results.items() 
                                      if isinstance(v, dict) and 'signal' in v and v['signal'] == 'neutral')
                    logger.debug(f"No trade for {symbol}: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral signals")
                    logger.debug(f"Need at least 4 signals in one direction to trade")
                
                logger.info(f"No clear trend for {symbol}, skipping")
    
    async def execute_trade(self, symbol, direction, analysis=None):
        """
        Execute a trade based on analysis.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('long' or 'short')
            analysis: Analysis results used to make the trading decision (optional)
        """
        try:
            if DEBUG_MODE:
                logger.debug(f"=== Starting trade execution for {symbol} ({direction}) ===")
            
            # Get current market price
            ticker = await self.api.get_ticker(symbol)
            current_price = float(ticker['last'])
            
            if DEBUG_MODE:
                logger.debug(f"Current market price for {symbol}: {current_price}")
            
            # Calculate position size based on account balance and current risk level
            account = await self.api.get_account_info()
            available_balance = float(account['available'])
            
            if DEBUG_MODE:
                logger.debug(f"Account balance: Available={available_balance}")
                logger.debug(f"Current risk level: {self.risk_level}, position size percent: {self.position_size_percent:.1%}")
                
                # If balance is too low, might be a reason trades aren't executing
                if available_balance < 10:
                    logger.warning(f"Low account balance: {available_balance} USDT. Trades might not execute due to insufficient funds.")
            
            # Use dynamic position sizing based on risk level
            position_size_usd = available_balance * self.position_size_percent
            
            # Limit maximum positions based on risk level
            if len(self.active_positions) >= self.max_positions:
                logger.warning(f"Maximum number of positions ({self.max_positions}) already reached. Skipping trade for {symbol}.")
                return
                
            # Additional safety: reduce position size further if drawdown is significant
            if self.current_drawdown > 0.10:  # Over 10% drawdown
                position_size_usd = position_size_usd * 0.8  # Reduce by additional 20%
                if DEBUG_MODE:
                    logger.debug(f"Applied additional position size reduction due to {self.current_drawdown:.2%} drawdown")
            
            # Ensure position size meets minimum requirements for Bitget
            min_notional = 5.0  # Minimum USD value for most exchanges
            min_amount = 1000.0  # Special for coins like 10000000AIDOGE with lots of zeros
            
            # Calculate quantity in coins
            position_quantity = position_size_usd / current_price
            
            # Check if position quantity is too small for certain coins (especially memecoins with many zeros)
            if "10000000" in symbol or "1000000" in symbol or "1000LUNC" in symbol:
                # For these special coins, ensure at least 1000 units
                if position_quantity < min_amount:
                    position_quantity = min_amount
                    position_size_usd = position_quantity * current_price
                    logger.info(f"Adjusted position size upward for {symbol} to meet minimum quantity requirements")
            
            if DEBUG_MODE:
                logger.debug(f"Position size: {position_size_usd} USDT ({self.position_size_percent*100:.1f}% of available balance)")
                logger.debug(f"Position quantity: {position_quantity} {symbol.split('/')[0]}")
                
                # Check minimum notional
                if position_size_usd < min_notional:
                    logger.warning(f"Position size ({position_size_usd} USDT) is below typical minimum notional requirement ({min_notional} USDT).")
            
            # Set leverage
            leverage_result = await self.api.set_leverage(symbol, self.leverage)
            
            if DEBUG_MODE and leverage_result:
                logger.debug(f"Leverage set to {self.leverage}x for {symbol}: {leverage_result}")
            
            # Calculate stop loss and take profit levels
            if direction == 'long':
                stop_loss = current_price * (1 - self.stop_loss_percent)
                take_profit = current_price * (1 + self.take_profit_percent)
            else:  # short
                stop_loss = current_price * (1 + self.stop_loss_percent)
                take_profit = current_price * (1 - self.take_profit_percent)
            
            if DEBUG_MODE:
                logger.debug(f"Stop loss: {stop_loss} ({self.stop_loss_percent*100}% from entry)")
                logger.debug(f"Take profit: {take_profit} ({self.take_profit_percent*100}% from entry)")
            
            # Execute the trade
            if DEBUG_MODE:
                logger.debug(f"Executing {direction} order: {symbol}, type=market, amount={position_quantity}, leverage={self.leverage}x")
            
            # Ensure minimum order size requirements are met
            # For SOL/USDT:USDT, the minimum is 0.1 SOL
            min_quantity = 0.1
            if symbol == "SOL/USDT:USDT" and position_quantity < min_quantity:
                logger.info(f"Adjusting order size for {symbol} from {position_quantity} to minimum {min_quantity}")
                position_quantity = min_quantity
            
            order = await self.api.create_order(
                symbol=symbol,
                type='market',
                side='buy' if direction == 'long' else 'sell',
                amount=position_quantity,
                leverage=self.leverage
            )
            
            if order and 'id' in order:
                if DEBUG_MODE:
                    logger.debug(f"Order successfully executed: {json.dumps(order, indent=2)}")
                
                # Store position details
                position = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_quantity,
                    'order_id': order['id'],
                    'leverage': self.leverage,
                    'open_time': time.time()
                }
                
                # Store analysis for learning from trade results
                if analysis:
                    position['entry_analysis'] = analysis
                    if DEBUG_MODE:
                        logger.debug(f"Stored analysis for {symbol} for adaptive learning")
                
                self.active_positions[symbol] = position
                
                # Send Telegram notification
                message = (
                    f"🔔 New {direction.upper()} position opened:\n"
                    f"Symbol: {symbol}\n"
                    f"Entry Price: {current_price:.4f}\n"
                    f"Stop Loss: {stop_loss:.4f}\n"
                    f"Take Profit: {take_profit:.4f}\n"
                    f"Leverage: {self.leverage}x\n"
                    f"Position Size: {position_size_usd:.2f} USD"
                )
                await send_telegram_message(message)
                
                logger.info(f"Trade executed for {symbol}: {direction} at {current_price}")
                return True
            else:
                logger.error(f"Failed to execute trade for {symbol}")
                if DEBUG_MODE:
                    logger.error(f"Order response: {order}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            if DEBUG_MODE:
                logger.error(f"Detailed error traceback: {traceback.format_exc()}")
            return False
    
    async def monitor_positions(self):
        """Monitor open positions and check if stop loss or take profit is hit."""
        if not self.active_positions:
            return
        
        positions_to_close = []
        
        for symbol, position in self.active_positions.items():
            try:
                # Get current price
                ticker = await self.api.get_ticker(symbol)
                current_price = float(ticker['last'])
                
                direction = position['direction']
                stop_loss = position['stop_loss']
                take_profit = position['take_profit']
                
                # Check if stop loss or take profit is hit
                if direction == 'long':
                    if current_price <= stop_loss:
                        logger.info(f"Stop loss hit for {symbol} long position")
                        positions_to_close.append((symbol, "stop loss", current_price))
                    elif current_price >= take_profit:
                        logger.info(f"Take profit hit for {symbol} long position")
                        positions_to_close.append((symbol, "take profit", current_price))
                else:  # short
                    if current_price >= stop_loss:
                        logger.info(f"Stop loss hit for {symbol} short position")
                        positions_to_close.append((symbol, "stop loss", current_price))
                    elif current_price <= take_profit:
                        logger.info(f"Take profit hit for {symbol} short position")
                        positions_to_close.append((symbol, "take profit", current_price))
                        
            except Exception as e:
                logger.error(f"Error monitoring position {symbol}: {str(e)}")
        
        # Close positions that hit stop loss or take profit
        for symbol, reason, price in positions_to_close:
            await self.close_position(symbol, reason, price)
    
    async def close_position(self, symbol, reason, price=None):
        """Close an open position."""
        try:
            if symbol not in self.active_positions:
                logger.warning(f"No active position found for {symbol}")
                return False
            
            position = self.active_positions[symbol]
            direction = position['direction']
            
            # Get current price if not provided
            if not price:
                ticker = await self.api.get_ticker(symbol)
                price = float(ticker['last'])
            
            # Create close order (opposite side of the open position)
            close_order = await self.api.create_order(
                symbol=symbol,
                type='market',
                side='sell' if direction == 'long' else 'buy',
                amount=position['position_size'],
                leverage=position['leverage']
            )
            
            if close_order and 'id' in close_order:
                # Calculate profit/loss
                entry_price = position['entry_price']
                pnl_percent = 0
                
                if direction == 'long':
                    pnl_percent = (price - entry_price) / entry_price * 100 * position['leverage']
                else:  # short
                    pnl_percent = (entry_price - price) / entry_price * 100 * position['leverage']
                
                # Send Telegram notification
                message = (
                    f"🔔 Position closed ({reason}):\n"
                    f"Symbol: {symbol}\n"
                    f"Direction: {direction.upper()}\n"
                    f"Entry Price: {entry_price:.4f}\n"
                    f"Close Price: {price:.4f}\n"
                    f"PnL: {pnl_percent:.2f}%\n"
                    f"Reason: {reason.capitalize()}"
                )
                await send_telegram_message(message)
                
                # Record in position history
                trade_record = {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl_percent': pnl_percent,
                    'open_time': position['open_time'],
                    'close_time': time.time(),
                    'reason': reason
                }
                self.position_history.append(trade_record)
                
                # Update adaptive strategy with trade result
                # This helps the strategy learn from past trades
                if 'entry_analysis' in position:
                    analysis_at_entry = position['entry_analysis']
                    # Update the adaptive strategy's weights based on this trade's outcome
                    trade_result = adaptive_strategy.record_trade_result(
                        symbol=symbol,
                        analysis=analysis_at_entry,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=price,
                        exit_reason=reason
                    )
                    
                    if DEBUG_MODE:
                        logger.debug(f"Updated adaptive strategy weights with trade result: {trade_result['profitable']}")
                        logger.debug(f"New signal weights: {adaptive_strategy.signal_weights}")
                
                # Update shared data for dashboard
                shared_data['completed_trades'].append(trade_record)
                shared_data['active_trades'] = {s: p for s, p in self.active_positions.items()}
                shared_data['new_activities'].append({
                    'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"Closed {direction} position for {symbol} with {pnl_percent:.2f}% PnL ({reason})",
                    'type': 'success' if pnl_percent > 0 else 'danger'
                })
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                logger.info(f"Position closed for {symbol} with PnL: {pnl_percent:.2f}%")
                return True
            else:
                logger.error(f"Failed to close position for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False
    
    async def run(self):
        """Main bot loop."""
        logger.info("Starting trading bot...")
        
        # Keep-alive for Replit
        keep_alive()
        
        # Initial portfolio check to establish baseline
        await self.check_portfolio_status()
        
        # Add initial portfolio stats to shared data for dashboard
        if self.initial_balance is not None:
            shared_data['new_activities'].append({
                'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': f"Initial portfolio balance: {self.initial_balance:.2f} USDT",
                'type': 'info'
            })
            
        # Initialize performance data for dashboard
        await self.initialize_performance_data()
        
        # Portfolio status monitoring interval (more frequent than full analysis)
        portfolio_check_interval = 120  # Check portfolio every 2 minutes
        last_portfolio_check = time.time()
        
        while True:
            try:
                current_time = time.time()
                
                # Check portfolio status more frequently than full analysis
                if current_time - last_portfolio_check >= portfolio_check_interval:
                    await self.check_portfolio_status()
                    last_portfolio_check = current_time
                    
                    # Update dashboard with portfolio information
                    shared_data['portfolio_balance'] = self.current_balance
                    shared_data['portfolio_drawdown'] = self.current_drawdown
                    shared_data['risk_level'] = self.risk_level
                    
                    # Update coin performance data for dashboard - using the function from extra_functions.py
                    if hasattr(self, 'update_coin_performance') and callable(self.update_coin_performance):
                        try:
                            coins_to_analyze = shared_data.get('top_volatile_coins', [])
                            # Ensure we also include important coins
                            important_coins = ["ETH/USDT:USDT", "SOL/USDT:USDT", "ADA/USDT:USDT", "XRP/USDT:USDT", "BTC/USDT:USDT"]
                            for coin in important_coins:
                                if coin not in coins_to_analyze:
                                    coins_to_analyze.append(coin)
                            
                            # Update coin performance data
                            if coins_to_analyze:
                                await self.update_coin_performance(coins_to_analyze)
                                logger.info(f"Updated performance data for {len(coins_to_analyze)} coins")
                        except Exception as e:
                            logger.error(f"Error updating coin performance: {str(e)}")
                
                # Analyze market for new opportunities
                await self.analyze_market()
                
                # Monitor existing positions
                await self.monitor_positions()
                
                # Wait before next analysis
                logger.info(f"Waiting {self.analysis_interval} seconds until next analysis")
                
                # Smaller sleep intervals to allow for more frequent portfolio checks
                sleep_interval = min(self.analysis_interval, portfolio_check_interval)
                for _ in range(int(self.analysis_interval / sleep_interval)):
                    await asyncio.sleep(sleep_interval)
                    
                    # Check positions during sleep period
                    if len(self.active_positions) > 0:
                        await self.monitor_positions()
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                if DEBUG_MODE:
                    logger.error(f"Detailed error traceback: {traceback.format_exc()}")
                await asyncio.sleep(60)  # Wait a minute before retrying on error

async def main():
    """Main entry point for the trading bot."""
    bot = TradingBot()
    await bot.run()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
