import os
import logging
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get logger
logger = logging.getLogger('trading_bot')

# Load Telegram bot token and chat ID from environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

async def send_telegram_message(message):
    """
    Send a message to a Telegram chat using the Telegram Bot API.
    
    Args:
        message (str): The message to send
        
    Returns:
        bool: True if the message was sent successfully, False otherwise
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram bot token or chat ID not set, skipping notification")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"  # Changed to HTML which is more reliable
    }
    
    try:
        # Debug log to help troubleshoot
        logger.info(f"Sending Telegram message to chat ID: {TELEGRAM_CHAT_ID}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                response_text = await response.text()
                if response.status == 200:
                    logger.info("Telegram notification sent successfully")
                    return True
                else:
                    logger.error(f"Failed to send Telegram notification. Status: {response.status}, Response: {response_text}")
                    # Try without parse_mode as a fallback
                    data.pop("parse_mode", None)
                    async with session.post(url, data=data) as fallback_response:
                        if fallback_response.status == 200:
                            logger.info("Telegram notification sent successfully (fallback without parse_mode)")
                            return True
                        else:
                            fallback_response_text = await fallback_response.text()
                            logger.error(f"Fallback also failed. Status: {fallback_response.status}, Response: {fallback_response_text}")
                            return False
    except Exception as e:
        logger.error(f"Error sending Telegram notification: {str(e)}")
        return False

async def test_telegram_connection():
    """Test the Telegram bot connection by sending a test message."""
    test_message = "🤖 Bitget Trading Bot is now online and ready for trading."
    success = await send_telegram_message(test_message)
    
    if success:
        logger.info("Telegram connection test successful")
    else:
        logger.warning("Telegram connection test failed")
    
    return success

# Function to allow simple synchronous sending (for non-async code)
def send_message_sync(message):
    """Synchronous wrapper for send_telegram_message."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Create a new loop for this call
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(send_telegram_message(message))
        finally:
            new_loop.close()
            asyncio.set_event_loop(loop)
    else:
        return loop.run_until_complete(send_telegram_message(message))
