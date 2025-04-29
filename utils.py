import os
import time
import logging
import threading
import http.server
import socketserver
from functools import wraps
from logging.handlers import RotatingFileHandler

def setup_logger():
    """Set up and configure logger."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure logger
    logger = logging.getLogger('trading_bot')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler for logs
    file_handler = RotatingFileHandler(
        'logs/trading_bot.log', 
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def retry(max_retries=3, delay=1):
    """Retry decorator for functions that might fail."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio  # Import here to avoid circular imports
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries < max_retries:
                        logger = logging.getLogger('trading_bot')
                        logger.warning(f"Retrying {func.__name__} after error: {str(e)}, attempt {retries}/{max_retries}")
                        await asyncio.sleep(delay * retries)  # Exponential backoff
                    else:
                        raise
        return wrapper
    return decorator

def keep_alive():
    """Start a simple HTTP server to keep the bot alive on platforms like Replit."""
    class KeepAliveHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Trading Bot is running!</h1></body></html>")
    
    def run_server():
        # Try different ports if the default is in use
        for port in range(8081, 8090):
            try:
                handler = KeepAliveHandler
                with socketserver.TCPServer(("", port), handler) as httpd:
                    logger = logging.getLogger('trading_bot')
                    logger.info(f"Starting keep-alive server on port {port}")
                    httpd.serve_forever()
                break  # If we get here, the server started successfully
            except OSError:
                logger = logging.getLogger('trading_bot')
                logger.warning(f"Port {port} already in use, trying next port")
                continue
    
    # Start the server in a separate thread
    thread = threading.Thread(target=run_server)
    thread.daemon = True
    thread.start()

def calculate_position_size(balance, risk_percent, entry_price, stop_loss):
    """
    Calculate position size based on risk management.
    
    Args:
        balance (float): Account balance
        risk_percent (float): Percentage of balance to risk (e.g., 0.01 for 1%)
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        
    Returns:
        float: Position size in quote currency
    """
    risk_amount = balance * risk_percent
    price_distance = abs(entry_price - stop_loss) / entry_price
    position_size = risk_amount / price_distance
    
    return position_size
