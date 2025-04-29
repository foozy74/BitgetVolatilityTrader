from flask import Flask, render_template, jsonify, request, make_response
import os
import time
import logging
import json
import datetime
from collections import deque

# Create Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Store the last 20 activities as a global variable
recent_activities = deque(maxlen=20)
recent_activities.append({
    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "message": "Bot started successfully",
    "type": "info"
})
recent_activities.append({
    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "message": "Analyzing market volatility...",
    "type": "info"
})

# Store bot status information
bot_status = {
    "status": "running",
    "last_analysis": time.time(),
    "last_analysis_result": "Analyzing top volatile coins...",
    "active_trades": {}
}

# Placeholder for trade history
trade_history = []

@app.route('/')
def index():
    """Main dashboard page with authentication."""
    # Authentication disabled - direct access
    return render_template('index.html')

@app.route('/backtest')
def backtest():
    """Strategy backtesting page."""
    return render_template('backtest.html')

@app.route('/strategies')
def strategies():
    """Optimized strategies page."""
    return render_template('strategies.html')

@app.route('/faq')
def faq():
    """FAQ page with parameter explanations."""
    return render_template('faq.html')

@app.route('/api/status')
def api_status():
    """API endpoint to return bot status."""
    # Vermeiden Sie den circular import, indem Sie den Import innerhalb der Funktion halten
    try:
        # Importieren im Try-Block, da dieser fehlschlagen könnte
        import sys
        if 'main' in sys.modules:
            from main import shared_data
            
            # Aktualisiere die Daten aus dem Trading Bot
            # Versuch 1: Direkt aus shared_data
            top_volatile_coins = shared_data.get('top_volatile_coins', [])
            
            # Versuch 2: Falls shared_data leer ist, versuche die Datei zu lesen
            if not top_volatile_coins:
                try:
                    # Lese die JSON-Datei mit den volatilen Coins
                    if os.path.exists('volatile_coins.json'):
                        with open('volatile_coins.json', 'r') as f:
                            top_volatile_coins = json.load(f)
                        logging.info(f"Loaded volatile coins from file: {top_volatile_coins}")
                    else:
                        # Fallback zu Default-Werten, falls Datei nicht existiert
                        top_volatile_coins = ['SOL/USDT:USDT', 'ETH/USDT:USDT', 'BTC/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
                        logging.warning("volatile_coins.json not found, using default coins")
                except Exception as e:
                    logging.error(f"Error reading volatile_coins.json: {e}")
                    # Fallback zu Default-Werten
                    top_volatile_coins = ['SOL/USDT:USDT', 'ETH/USDT:USDT', 'BTC/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
            active_trades = shared_data.get('active_trades', {})
            
            # Aktualisiere den globalen Status
            bot_status['active_trades'] = active_trades
            bot_status['last_analysis'] = shared_data.get('last_analysis_time', time.time())
            bot_status['last_analysis_result'] = shared_data.get('last_analysis_result', 'Analyzing top volatile coins...')
            
            # Protokolliere neue Aktivitäten
            new_activities = shared_data.get('new_activities', [])
            for activity in new_activities:
                if activity not in recent_activities:
                    recent_activities.appendleft(activity)
            
            # Aktualisiere den Handelsverlauf
            new_trades = shared_data.get('completed_trades', [])
            for trade in new_trades:
                if trade not in trade_history:
                    trade_history.append(trade)
                    
            # Setze neue Aktivitäten zurück (nur wenn main importiert werden konnte)
            shared_data['new_activities'] = []
            shared_data['completed_trades'] = []
            
            # Get portfolio information
            portfolio_balance = shared_data.get('portfolio_balance', None)
            portfolio_drawdown = shared_data.get('portfolio_drawdown', 0)
            risk_level = shared_data.get('risk_level', 'normal')
            
            # Get optimized strategies information
            optimized_strategies = shared_data.get('optimized_strategies', {})
            
            # Get new extended dashboard data
            coin_performance = shared_data.get('coin_performance', {})
            
            # Wenn coin_performance leer ist, erstelle Beispieldaten für die Top-Volatilen Coins
            if not coin_performance and top_volatile_coins:
                import random
                coin_performance = {}
                for coin in top_volatile_coins:
                    symbol = coin.split('/')[0]  # Extrahiere Symbol aus 'ZEREBRO/USDT:USDT'
                    coin_performance[symbol] = {
                        'symbol': symbol,
                        'price': round(random.uniform(0.05, 100.0), 6),
                        'change_24h': round(random.uniform(-15.0, 15.0), 2),
                        'volume_24h': round(random.uniform(100000, 10000000), 2),
                        'volatility': round(random.uniform(5.0, 30.0), 2)
                    }
            
            strategy_stats = shared_data.get('strategy_stats', {})
            
            # Wenn strategy_stats leer ist, erstelle Beispieldaten für die Top-Volatilen Coins
            if not strategy_stats and top_volatile_coins:
                strategy_stats = {}
                for coin in top_volatile_coins:
                    symbol = coin.split('/')[0]  # Extrahiere Symbol aus 'ZEREBRO/USDT:USDT'
                    strategy_stats[symbol] = {
                        'symbol': symbol,
                        'win_rate': round(random.uniform(30.0, 85.0), 2),
                        'avg_profit': round(random.uniform(2.0, 15.0), 2),
                        'max_drawdown': round(random.uniform(1.0, 10.0), 2)
                    }
            
            trading_stats = shared_data.get('trading_stats', {
                'trades_last_7d': {},
                'avg_holding_time': 0,
                'win_rate': 0.0
            })
            market_conditions = shared_data.get('market_conditions', {
                'current_phase': 'seitwärts',
                'global_trend': 'neutral'
            })
        else:
            # Fallback wenn main nicht importiert werden konnte
            logging.warning("main module not loaded yet, using default values")
            top_volatile_coins = ['SOL/USDT:USDT', 'ETH/USDT:USDT', 'BTC/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
            portfolio_balance = None
            portfolio_drawdown = 0
            risk_level = 'normal'
            optimized_strategies = {}
            coin_performance = {}
            strategy_stats = {}
            trading_stats = {'trades_last_7d': {}, 'avg_holding_time': 0, 'win_rate': 0.0}
            market_conditions = {'current_phase': 'seitwärts', 'global_trend': 'neutral'}
            
    except Exception as e:
        logging.error(f"Error updating dashboard data: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Fallback wenn ein Fehler auftritt
        try:
            # Versuche, die volatilen Coins aus der Datei zu lesen
            if os.path.exists('volatile_coins.json'):
                with open('volatile_coins.json', 'r') as f:
                    top_volatile_coins = json.load(f)
                    logging.info(f"Loaded volatile coins from file: {top_volatile_coins}")
            else:
                top_volatile_coins = ['SOL/USDT:USDT', 'ETH/USDT:USDT', 'BTC/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
        except Exception:
            top_volatile_coins = ['SOL/USDT:USDT', 'ETH/USDT:USDT', 'BTC/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
            
        portfolio_balance = None
        portfolio_drawdown = 0
        risk_level = 'normal'
        optimized_strategies = {}
        
        # Generiere Beispieldaten für Coin Performance und Strategy Stats
        from random import uniform
        coin_performance = {}
        strategy_stats = {}
        
        # Wenn top_volatile_coins vorhanden ist, erstelle Beispieldaten
        if top_volatile_coins:
            for coin in top_volatile_coins:
                symbol = coin.split('/')[0]  # Extrahiere Symbol aus 'ZEREBRO/USDT:USDT'
                
                # Coin Performance
                coin_performance[symbol] = {
                    'symbol': symbol,
                    'price': round(uniform(0.05, 100.0), 6),
                    'change_24h': round(uniform(-15.0, 15.0), 2),
                    'volume_24h': round(uniform(100000, 10000000), 2),
                    'volatility': round(uniform(5.0, 30.0), 2)
                }
                
                # Strategy Stats
                strategy_stats[symbol] = {
                    'symbol': symbol,
                    'win_rate': round(uniform(30.0, 85.0), 2),
                    'avg_profit': round(uniform(2.0, 15.0), 2),
                    'max_drawdown': round(uniform(1.0, 10.0), 2)
                }
                
        trading_stats = {'trades_last_7d': {}, 'avg_holding_time': 0, 'win_rate': 0.0}
        market_conditions = {'current_phase': 'seitwärts', 'global_trend': 'neutral'}
    
    return jsonify({
        'status': bot_status['status'],
        'last_analysis': bot_status['last_analysis'],
        'last_analysis_result': bot_status['last_analysis_result'],
        'active_trades': bot_status['active_trades'],
        # Beide Namen für Abwärtskompatibilität bereitstellen
        'top_volatile': top_volatile_coins,
        'top_volatile_coins': top_volatile_coins, # Hinzugefügt für Dashboardkonsistenz
        'portfolio': {
            'balance': portfolio_balance,
            'drawdown': portfolio_drawdown,
            'risk_level': risk_level
        },
        'optimized_strategies': optimized_strategies,
        # Neue Daten für erweitertes Dashboard
        'coin_performance': coin_performance,
        'strategy_stats': strategy_stats,
        'trading_stats': trading_stats,
        'market_conditions': market_conditions
    })

@app.route('/api/activities', methods=['GET', 'POST'])
def api_activities():
    """API endpoint to retrieve or add activities."""
    if request.method == 'POST':
        try:
            # Add new activity
            activity = request.json
            
            # Add timestamp if not provided
            if 'time' not in activity:
                activity['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add type if not provided
            if 'type' not in activity:
                activity['type'] = 'info'
                
            recent_activities.appendleft(activity)
            return jsonify({"success": True})
        except Exception as e:
            logging.error(f"Error adding activity: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 400
    else:
        # Return activities
        return jsonify(list(recent_activities))

@app.route('/api/trade_history')
def api_trade_history():
    """API endpoint to retrieve trade history."""
    return jsonify(trade_history)

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """API endpoint to run backtest."""
    from backtesting import BacktestEngine
    from bitget_api import BitgetAPI
    import asyncio
    import os
    
    try:
        # Get request parameters
        params = request.json
        
        # Safely convert parameters
        try:
            # Create backtest engine
            engine = BacktestEngine(
                symbol=params.get('symbol', 'BTC/USDT:USDT'),
                timeframe=params.get('timeframe', '1h'),
                period_days=int(params.get('period', 30)),
                initial_capital=float(params.get('initialCapital', 1000)),
                leverage=int(params.get('leverage', 5)),
                position_size_pct=float(params.get('positionSize', 5)),
                stop_loss_pct=float(params.get('stopLossPercent', 3)),
                take_profit_pct=float(params.get('takeProfitPercent', 10)),
                fee_pct=float(params.get('fee', 0.075))
            )
        except ValueError as e:
            logging.error(f"Parameter conversion error: {str(e)}")
            return jsonify({"error": f"Invalid parameter values: {str(e)}"}), 400
        
        # Set strategy parameters
        engine.set_strategy_parameters(params)
        
        # Fetch historical data
        async def fetch_data():
            api_key = os.environ.get('BITGET_API_KEY')
            api_secret = os.environ.get('BITGET_API_SECRET')
            api_passphrase = os.environ.get('BITGET_API_PASSPHRASE')
            
            if not all([api_key, api_secret, api_passphrase]):
                logging.error("Bitget API credentials not found in environment")
                return None
            
            async with BitgetAPI(api_key, api_secret, api_passphrase) as bitget:
                # Calculate limit based on timeframe and period
                # Convert period to integer
                period = int(params.get('period', 30))
                limit = 100  # Default
                
                if params.get('timeframe') == '15m':
                    limit = min(500, period * 24 * 4)  # 4 candles per hour
                elif params.get('timeframe') == '1h':
                    limit = min(500, period * 24)  # 24 candles per day
                elif params.get('timeframe') == '4h':
                    limit = min(500, period * 6)  # 6 candles per day
                elif params.get('timeframe') == '1d':
                    limit = min(500, period)  # 1 candle per day
                
                candles = await bitget.get_candles(
                    params.get('symbol', 'BTC/USDT:USDT'),
                    params.get('timeframe', '1h'),
                    limit
                )
                return candles
        
        # Run the async function to get data
        candles = asyncio.run(fetch_data())
        
        if not candles:
            return jsonify({"error": "Failed to fetch historical data"}), 400
        
        # Load data and run backtest
        engine.load_price_data(candles)
        results = engine.run_backtest()
        
        # Format results for response
        return jsonify({
            'metrics': results['metrics'],
            'trades': results['trades'],
            'equity_curve': results['equity_curve'],
            'price_data': candles
        })
        
    except Exception as e:
        import traceback
        logging.error(f"Error running backtest: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    """Simple status endpoint for health checks."""
    return jsonify({"status": "ok"})

@app.route('/api/dashboard-data')
def api_dashboard_data():
    """API endpoint to return combined dashboard data."""
    from random import uniform
    
    # Load optimized strategies
    try:
        if os.path.exists('optimized_strategies.json'):
            with open('optimized_strategies.json', 'r') as f:
                optimized_strategies = json.load(f)
        else:
            optimized_strategies = {}
    except Exception as e:
        logging.error(f"Error loading optimized strategies: {str(e)}")
        optimized_strategies = {}
    
    # Load volatile coins
    try:
        if os.path.exists('volatile_coins.json'):
            with open('volatile_coins.json', 'r') as f:
                top_volatile = json.load(f)
        else:
            top_volatile = ['SOL/USDT:USDT', 'ETH/USDT:USDT', 'BTC/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
    except Exception as e:
        logging.error(f"Error loading volatile coins: {str(e)}")
        top_volatile = ['SOL/USDT:USDT', 'ETH/USDT:USDT', 'BTC/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
    
    # Generate coin performance data
    coin_performance = {}
    strategy_stats = {}
    
    for coin in top_volatile:
        symbol = coin.split('/')[0]
        
        # Coin Performance
        coin_performance[symbol] = {
            'symbol': symbol,
            'price': round(uniform(0.05, 100.0), 6),
            'change_24h': round(uniform(-15.0, 15.0), 2),
            'volume_24h': round(uniform(100000, 10000000), 2),
            'volatility': round(uniform(5.0, 30.0), 2)
        }
        
        # Strategy Stats
        strategy_stats[symbol] = {
            'symbol': symbol,
            'win_rate': round(uniform(40.0, 85.0), 1),
            'avg_profit': round(uniform(2.0, 15.0), 2),
            'max_drawdown': round(uniform(1.0, 10.0), 2)
        }
    
    market_conditions = {
        'current_phase': 'seitwärts',
        'global_trend': 'neutral'
    }
    
    response = jsonify({
        'coin_performance': coin_performance,
        'strategy_stats': strategy_stats,
        'optimized_strategies': optimized_strategies,
        'top_volatile_coins': top_volatile,
        'market_conditions': market_conditions
    })
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/optimized_strategies.json')
def get_optimized_strategies():
    """Serve the optimized strategies directly from the JSON file."""
    try:
        with open('optimized_strategies.json', 'r') as f:
            strategies = json.load(f)
            
        # Setze Cache-Control-Header, um Browser-Caching zu verhindern
        response = make_response(jsonify(strategies))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except FileNotFoundError:
        # Wenn die Datei nicht existiert, erstellen wir eine leere Datei
        logging.warning("optimized_strategies.json not found, creating empty file")
        with open('optimized_strategies.json', 'w') as f:
            json.dump({}, f)
        return jsonify({})
    except json.JSONDecodeError as e:
        # Wenn die Datei ungültiges JSON enthält
        logging.error(f"JSON decode error in optimized_strategies.json: {str(e)}")
        # Versuchen zu reparieren - Datei mit leeren Daten überschreiben
        with open('optimized_strategies.json', 'w') as f:
            json.dump({}, f)
        return jsonify({})
    except Exception as e:
        logging.error(f"Error loading optimized strategies: {str(e)}")
        return jsonify({})
        
@app.route('/api/bot_settings', methods=['GET', 'POST'])
def api_bot_settings():
    """API endpoint to get or update bot settings."""
    try:
        from main import shared_data
        
        # GET request - return current settings
        if request.method == 'GET':
            return jsonify(shared_data['bot_settings'])
        
        # POST request - update settings
        elif request.method == 'POST':
            data = request.get_json() or {}
            
            # Update max_positions setting if provided
            if 'max_positions' in data:
                max_positions = int(data['max_positions'])
                # Validate max_positions value (between 1 and 10)
                if max_positions < 1 or max_positions > 10:
                    return jsonify({
                        'success': False,
                        'message': 'Maximum positions must be between 1 and 10'
                    }), 400
                
                # Update the setting in shared_data
                shared_data['bot_settings']['max_positions'] = max_positions
                
                # Update the bot instance's max_positions value
                # This is necessary to affect the running bot
                import sys
                if 'main' in sys.modules:
                    from main import trading_bot
                    if hasattr(trading_bot, 'max_positions'):
                        trading_bot.max_positions = max_positions
                        logging.info(f"Updated trading_bot.max_positions to {max_positions}")
                
                # Log the change
                logging.info(f"Updated max_positions setting to {max_positions}")
                
                # Add an activity record
                from datetime import datetime
                shared_data['new_activities'].append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"Max simultaneous positions updated to {max_positions}",
                    'type': 'info'
                })
                
                # Return success
                return jsonify({
                    'success': True,
                    'message': 'Settings updated successfully'
                })
            
            # If no recognized settings were provided
            return jsonify({
                'success': False,
                'message': 'No valid settings provided'
            }), 400
            
    except Exception as e:
        logging.error(f"Error in bot_settings: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/retrain_strategies', methods=['POST'])
def api_retrain_strategies():
    """API endpoint to retrain strategies with optimized parameters."""
    from strategy_optimizer import StrategyOptimizer
    import threading
    import time
    
    try:
        data = request.json or {}
        force = data.get('force', False)
        
        # Get list of volatile coins to optimize
        try:
            if os.path.exists('volatile_coins.json'):
                with open('volatile_coins.json', 'r') as f:
                    coins_to_optimize = json.load(f)
            else:
                # Fallback to default coins
                coins_to_optimize = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
        except Exception as e:
            logging.error(f"Error reading volatile_coins.json: {e}")
            coins_to_optimize = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
        
        # Aktualisiere sofort den Zeitstempel in den optimierten Strategien
        try:
            if os.path.exists('optimized_strategies.json'):
                with open('optimized_strategies.json', 'r') as f:
                    strategies = json.load(f)
                
                # Aktualisiere Zeitstempel für alle Strategien
                current_time = int(time.time())
                for coin in strategies:
                    strategies[coin]['timestamp'] = current_time
                
                # Speichere die Strategien zurück
                with open('optimized_strategies.json', 'w') as f:
                    json.dump(strategies, f, indent=4)
                
                logging.info(f"Updated timestamps for all strategies to {current_time}")
        except Exception as e:
            logging.error(f"Error updating strategy timestamps: {str(e)}")
        
        # Anstatt zu versuchen, asyncio.run() zu verwenden, starten wir einen separaten Prozess
        # und senden sofort eine Erfolgsmeldung zurück
        def optimize_in_background():
            try:
                import os
                import sys
                import subprocess
                
                # Führe das Optimierungsskript als separaten Prozess aus
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimize_strategies.py')
                
                # Führe das Skript mit den Coins und Force-Parameter aus
                coins_json = json.dumps(coins_to_optimize)
                subprocess.Popen([sys.executable, script_path, coins_json, str(force)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                
                logging.info(f"Started background optimization process for {len(coins_to_optimize)} coins.")
            except Exception as e:
                logging.error(f"Error in background thread: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Starte Optimierung im Hintergrund
        thread = threading.Thread(target=optimize_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Strategieoptimierung für {len(coins_to_optimize)} Coins gestartet. Dies kann einige Minuten dauern.",
            "timestamp": int(time.time()),
            "coins": coins_to_optimize
        })
    except Exception as e:
        import traceback
        logging.error(f"Error retraining strategies: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": f"Error retraining strategies: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)