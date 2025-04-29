#!/usr/bin/env python3
import asyncio
import json
import sys
import os
import logging
import time
from datetime import datetime
from strategy_optimizer import StrategyOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Lade die zu optimierenden Coins
        if len(sys.argv) > 1:
            coins_json = sys.argv[1]
            coins = json.loads(coins_json)
        else:
            # Versuche, aus der Datei zu laden
            if os.path.exists('volatile_coins.json'):
                with open('volatile_coins.json', 'r') as f:
                    coins = json.load(f)
            else:
                coins = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'ADA/USDT:USDT', 'XRP/USDT:USDT']
        
        # Bestimme, ob Optimierung erzwungen werden soll
        force = False
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == 'true'
        
        # Aktualisiere den Zeitstempel in den Strategien vor der Optimierung
        current_time = int(time.time())
        try:
            if os.path.exists('optimized_strategies.json'):
                with open('optimized_strategies.json', 'r') as f:
                    strategies = json.load(f)
                    
                # Aktualisiere nur den Zeitstempel
                for coin in strategies:
                    strategies[coin]['timestamp'] = current_time
                    
                # Speichere die aktualisierten Strategien zurück
                with open('optimized_strategies.json', 'w') as f:
                    json.dump(strategies, f, indent=4)
                    
                logger.info(f"Updated timestamps for all strategies to {current_time}")
        except Exception as e:
            logger.error(f"Error updating strategy timestamps: {str(e)}")
        
        # Erstelle Optimizer und führe Optimierung durch
        optimizer = StrategyOptimizer()
        await optimizer.retrain_strategies(coins, force)
        
        logger.info(f"Successfully optimized strategies for {len(coins)} coins.")
    except Exception as e:
        logger.error(f"Error optimizing strategies: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(main())