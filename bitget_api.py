import hmac
import base64
import hashlib
import time
import json
import logging
import asyncio
import ccxt.async_support as ccxt
from urllib.parse import urlencode

logger = logging.getLogger('trading_bot')

class BitgetAPI:
    def __init__(self, api_key, api_secret, api_passphrase):
        """
        Initialize the Bitget API wrapper.
        
        Args:
            api_key (str): Bitget API key
            api_secret (str): Bitget API secret
            api_passphrase (str): Bitget API passphrase
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        
        # Initialize ccxt exchange object
        self.exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'password': api_passphrase,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # Use futures/swap markets
            }
        })
        
        logger.info("Bitget API client initialized")
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exchange.close()
    
    async def _sign_request(self, method, endpoint, params=None):
        """Sign a request using Bitget authentication method."""
        timestamp = str(int(time.time() * 1000))
        
        if params is None:
            params = {}
        
        if method == 'GET' and params:
            endpoint = f"{endpoint}?{urlencode(params)}"
            message = timestamp + method + endpoint
        else:
            message = timestamp + method + endpoint + (json.dumps(params) if params else '')
        
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        headers = {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.api_passphrase,
            'Content-Type': 'application/json'
        }
        
        return headers
    
    async def get_futures_markets(self):
        """Get all available futures markets."""
        try:
            markets = await self.exchange.load_markets()
            futures_symbols = [symbol for symbol in markets.keys() if 'USDT' in symbol and self.exchange.markets[symbol]['swap']]
            return futures_symbols
        except Exception as e:
            logger.error(f"Error getting futures markets: {str(e)}")
            return []
        
    async def get_ticker(self, symbol):
        """Get ticker information for a symbol."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            # Return a minimal ticker dictionary to avoid None errors
            return {
                'symbol': symbol,
                'last': 0,
                'bid': 0,
                'ask': 0,
                'high': 0,
                'low': 0,
                'volume': 0,
                'timestamp': int(time.time() * 1000)
            }
    
    async def get_candles(self, symbol, timeframe='1h', limit=100):
        """
        Get candlestick data for a symbol.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit (int): Number of candles to fetch
            
        Returns:
            list: List of candles [timestamp, open, high, low, close, volume]
        """
        try:
            candles = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return candles
        except Exception as e:
            logger.error(f"Error getting candles for {symbol}: {str(e)}")
            return []
    
    async def get_account_info(self):
        """Get account balance information."""
        try:
            balance = await self.exchange.fetch_balance()
            return {
                'total': balance['total']['USDT'] if 'USDT' in balance['total'] else 0,
                'used': balance['used']['USDT'] if 'USDT' in balance['used'] else 0,
                'free': balance['free']['USDT'] if 'USDT' in balance['free'] else 0,
                'available': balance['free']['USDT'] if 'USDT' in balance['free'] else 0
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {'total': 0, 'used': 0, 'free': 0, 'available': 0}
    
    async def set_leverage(self, symbol, leverage):
        """Set leverage for a trading pair."""
        try:
            # Use crossed margin mode to match account settings
            params = {
                'marginMode': 'crossed',  # Use crossed margin mode to match account settings
                'productType': 'USDT-FUTURES'
            }
            
            result = await self.exchange.set_leverage(leverage, symbol, params=params)
            logger.info(f"Leverage set for {symbol}: {leverage}x")
            return result
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {str(e)}")
            return None
    
    async def create_order(self, symbol, type, side, amount, price=None, leverage=5):
        """
        Create a new order.
        
        Args:
            symbol (str): Trading pair symbol
            type (str): Order type (market, limit)
            side (str): Order side (buy, sell)
            amount (float): Order amount
            price (float, optional): Order price (required for limit orders)
            leverage (int): Leverage to use
            
        Returns:
            dict: Order information
        """
        try:
            # Make sure leverage is set before placing order
            await self.set_leverage(symbol, leverage)
            
            # Create the order with the correct margin mode
            order_params = {
                'marginMode': 'crossed',  # Use crossed margin mode to match account settings
                'productType': 'USDT-FUTURES',  # Specify futures product type
                'positionSide': 'long' if side == 'buy' else 'short'  # Specify whether it's a long or short position
            }
            
            if type.lower() == 'market':
                order = await self.exchange.create_market_order(symbol, side, amount, params=order_params)
            else:
                if price is None:
                    raise ValueError("Price is required for limit orders")
                order = await self.exchange.create_limit_order(symbol, side, amount, price, params=order_params)
                
            logger.info(f"Order created: {symbol} {type} {side} {amount} at price {price if price else 'market'}")
            return order
        except Exception as e:
            logger.error(f"Error creating order: {str(e)}")
            return None
    
    async def cancel_order(self, order_id, symbol):
        """Cancel an existing order."""
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} for {symbol} cancelled")
            return result
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return None
    
    async def get_position(self, symbol):
        """Get current position for a symbol."""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            position = next((p for p in positions if p['symbol'] == symbol), None)
            return position
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            # Return empty position to avoid None errors
            return {
                'symbol': symbol,
                'side': 'flat',
                'entryPrice': 0,
                'amount': 0,
                'leverage': 0
            }
    
    async def get_open_orders(self, symbol=None):
        """Get all open orders, optionally filtered by symbol."""
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"Error getting open orders: {str(e)}")
            return []
    
    async def close(self):
        """Close the exchange connection."""
        await self.exchange.close()
