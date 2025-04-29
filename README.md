# Bitget Futures Trading Bot

This is an automated trading bot for Bitget Futures that analyzes market volatility, identifies trading opportunities using technical indicators, executes trades with leverage, and sends real-time notifications via Telegram.

## Features

- Connects to Bitget Futures API to access market data and execute trades
- Analyzes the market every 10 minutes to find the top 5 most volatile altcoins
- Uses multiple technical indicators (RSI, MACD, OBV, VWAP, SMA, Bollinger Bands) to identify trading opportunities
- Automatically determines trade direction (long/short) based on indicator analysis
- Executes trades with 5x leverage
- Implements soft stop-loss (3%) and take-profit (10%) levels
- Sends real-time trade notifications via Telegram
- Runs in an endless loop, continuously monitoring the market and open positions

## Requirements

- Python 3.9+
- ccxt
- pandas
- numpy
- pandas-ta
- python-telegram-bot
- python-dotenv
- aiohttp

## Setup

1. Clone this repository:

```bash
git clone <repository-url>
cd bitget-trading-bot
