import pandas as pd
import numpy as np

# Define NaN value (missing in some numpy versions)
np.NaN = float('nan')

import pandas_ta as ta

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index (RSI)."""
    try:
        # Calculate RSI using pandas_ta
        rsi_result = ta.rsi(df['close'], length=period)
        
        # Fix for "truth value of Series is ambiguous" error
        valid_rsi = False
        if rsi_result is not None:
            if not isinstance(rsi_result, pd.Series):
                valid_rsi = True
            elif not rsi_result.empty:
                valid_rsi = True
                
        if valid_rsi:
            df['rsi'] = rsi_result
        else:
            raise ValueError("RSI calculation returned None or empty")
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error calculating RSI: {str(e)}")
        
        # Manual RSI calculation as fallback
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    try:
        # Calculate MACD using pandas_ta
        macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)
        # Fix for "truth value of Series is ambiguous" error - check if macd is valid before concat
        valid_macd = False
        if macd is not None:
            if not isinstance(macd, pd.Series):
                valid_macd = True
            elif not macd.empty:
                valid_macd = True
                
        if valid_macd:
            df = pd.concat([df, macd], axis=1)
        else:
            # Fallback calculation if pandas_ta returns None or empty Series
            raise ValueError("MACD calculation returned None or empty")
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error calculating MACD: {str(e)}")
        # Simple fallback MACD calculation
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['MACD_12_26_9'] = ema_fast - ema_slow
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=signal, adjust=False).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
    
    return df

def calculate_obv(df):
    """Calculate On-Balance Volume (OBV)."""
    try:
        # Calculate OBV using pandas_ta
        obv_result = ta.obv(df['close'], df['volume'])
        
        # Fix for "truth value of Series is ambiguous" error
        # Check if result is valid before using it
        if obv_result is not None and not (isinstance(obv_result, pd.Series) and obv_result.empty):
            df['obv'] = obv_result
        else:
            raise ValueError("OBV calculation returned None or empty")
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error calculating OBV: {str(e)}")
        
        # Manual OBV calculation as fallback
        obv = 0
        obv_values = []
        
        for i in range(len(df)):
            if i > 0:
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv += df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv -= df['volume'].iloc[i]
            obv_values.append(obv)
            
        df['obv'] = obv_values
        
    return df

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price (VWAP)."""
    try:
        # Make sure we have a datetime index
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            # Convert timestamp column to datetime if available
            if 'timestamp' in df_copy.columns:
                df_copy['datetime'] = pd.to_datetime(df_copy['timestamp'], unit='ms')
                df_copy.set_index('datetime', inplace=True)
            
        # Calculate VWAP using pandas_ta (only works with datetime index)
        # Attempt to use pandas_ta's VWAP calculation, but with robust error handling
        try:
            vwap = ta.vwap(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'])
            
            # Safe check for vwap - avoid direct truth value testing of Series
            valid_vwap = False
            if vwap is not None:
                if not isinstance(vwap, pd.Series):
                    valid_vwap = True
                elif hasattr(vwap, 'empty') and not vwap.empty:
                    valid_vwap = True
            
            if valid_vwap:
                try:
                    vwap.reset_index(drop=True, inplace=True)
                    df_vwap = pd.DataFrame(vwap)
                    df_vwap.reset_index(drop=True, inplace=True)
                    
                    # Save the original index
                    orig_index = df.index.copy()
                    
                    # Reset index for safe concatenation
                    df_reset = df.reset_index(drop=True)
                    df_result = pd.concat([df_reset, df_vwap], axis=1)
                    
                    # Restore original index if needed
                    if not df.index.equals(pd.RangeIndex(len(df))):
                        df_result.index = orig_index
                    
                    df = df_result
                except Exception as e:
                    import logging
                    logger = logging.getLogger('trading_bot')
                    logger.error(f"Error merging pandas_ta VWAP: {str(e)}")
                    # Will fall back to the manual calculation below
        except Exception as e:
            import logging
            logger = logging.getLogger('trading_bot')
            logger.error(f"pandas_ta VWAP calculation failed: {str(e)}")
            # Will fall back to the manual calculation below
        
        # Add a standard VWAP calculation which is more reliable
        typical_price = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        vol_cumsum = df_copy['volume'].cumsum()
        
        # Safe handling to avoid division by zero
        if not vol_cumsum.empty and vol_cumsum.iloc[-1] > 0:
            df_copy['VWAP_D'] = (typical_price * df_copy['volume']).cumsum() / vol_cumsum
        else:
            # If volume is zero, VWAP is equal to the typical price
            df_copy['VWAP_D'] = typical_price
        
        # Copy VWAP_D to the original dataframe
        df['VWAP_D'] = df_copy['VWAP_D'].values
        
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error calculating VWAP: {str(e)}")
        
        # Ultra safe fallback - manually calculate a simple VWAP
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vol_sum = df['volume'].sum()
            
            if vol_sum > 0:
                df['VWAP_D'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            else:
                df['VWAP_D'] = typical_price
        except Exception as e2:
            logger.error(f"Fallback VWAP calculation also failed: {str(e2)}")
            # Last resort - set VWAP to close price
            df['VWAP_D'] = df['close']
    
    return df

def calculate_sma(df, period=20):
    """Calculate Simple Moving Average (SMA)."""
    try:
        # Calculate SMA using pandas_ta
        sma_result = ta.sma(df['close'], length=period)
        
        # Fix for "truth value of Series is ambiguous" error
        if sma_result is not None and not (isinstance(sma_result, pd.Series) and sma_result.empty):
            df[f'sma_{period}'] = sma_result
        else:
            raise ValueError(f"SMA calculation returned None or empty for period {period}")
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error calculating SMA: {str(e)}")
        
        # Manual SMA calculation as fallback
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    
    return df

def calculate_ema(df, period=20):
    """Calculate Exponential Moving Average (EMA)."""
    column_name = f'ema_{period}'
    
    try:
        # Calculate EMA using pandas_ta
        ema_result = ta.ema(df['close'], length=period)
        
        # Fix for "truth value of Series is ambiguous" error
        if ema_result is not None and not (isinstance(ema_result, pd.Series) and ema_result.empty):
            df[column_name] = ema_result
        else:
            raise ValueError(f"EMA calculation returned None or empty for period {period}")
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error calculating EMA: {str(e)}")
        
        # Manual EMA calculation as fallback
        df[column_name] = df['close'].ewm(span=period, adjust=False).mean()
    
    return df

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    # Fix for "truth value of Series is ambiguous" error
    # Ensure we're working with proper types and handle Series objects appropriately
    try:
        bbands = ta.bbands(df['close'], length=period, std=std_dev)
        # Check if bbands is valid before concatenation
        if bbands is not None and not (isinstance(bbands, pd.Series) and bbands.empty):
            df = pd.concat([df, bbands], axis=1)
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        # Calculate simple Bollinger Bands as fallback
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['BBU_20_2.0'] = sma + (std_dev * std)
        df['BBM_20_2.0'] = sma
        df['BBL_20_2.0'] = sma - (std_dev * std)
    
    return df

def analyze_coin(df):
    """Analyze a coin using multiple technical indicators."""
    # Clean and prepare data
    df = df.copy()
    
    try:
        # Calculate indicators
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = calculate_obv(df)
        df = calculate_vwap(df)
        df = calculate_sma(df, 20)
        df = calculate_sma(df, 50)
        df = calculate_ema(df, 20)
        df = calculate_bollinger_bands(df)
        
        # Get the latest data for analysis
        latest_data = df.iloc[-1].copy()
        
        # Use safe get for values with fallbacks
        def safe_get(row, key, default=0):
            try:
                # First check if key exists in the row
                if key not in row:
                    return float(default)
                
                # Now handle NaN values - without direct truth value testing
                if isinstance(row[key], (pd.Series, np.ndarray)):
                    if pd.isna(row[key]).all():
                        return float(default)
                elif pd.isna(row[key]):
                    return float(default)
                
                # Get the value, handling both scalar and vector cases
                val = row[key]
                
                # If it's a Series or array, safely extract a scalar
                if isinstance(val, (pd.Series, np.ndarray)):
                    if hasattr(val, 'size') and val.size == 0:
                        return float(default)
                    elif hasattr(val, 'iloc'):
                        return float(val.iloc[0])
                    elif len(val) > 0:
                        return float(val[0])
                    else:
                        return float(default)
                elif isinstance(val, (int, float, np.number)):
                    # It's already a scalar numeric value
                    return float(val)
                else:
                    # Try to convert other types (like strings)
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return float(default)
            except Exception as e:
                import logging
                logger = logging.getLogger('trading_bot')
                logger.error(f"Error in safe_get with key {key}: {str(e)}")
                return float(default)
        
        # Analyze RSI - explicit float casting and comparison
        rsi_val = safe_get(latest_data, 'rsi', 50.0)
        rsi_float = float(rsi_val)  # Ensure it's a float scalar
        if rsi_float < 30.0:
            rsi_signal = 'buy'
        elif rsi_float > 70.0:
            rsi_signal = 'sell'
        else:
            rsi_signal = 'neutral'
        
        # Analyze MACD - explicit float casting and comparison
        macd_float = float(safe_get(latest_data, 'MACD_12_26_9', 0.0))
        macd_signal_float = float(safe_get(latest_data, 'MACDs_12_26_9', 0.0))
        if macd_float > macd_signal_float:
            macd_signal = 'buy'
        elif macd_float < macd_signal_float:
            macd_signal = 'sell'
        else:
            macd_signal = 'neutral'
        
        # Analyze OBV - explicit float casting and comparison
        obv_float = float(safe_get(latest_data, 'obv', 0.0))
        obv_prev_float = float(safe_get(df.iloc[-2], 'obv', 0.0) if len(df) > 1 else 0.0)
        if obv_float > obv_prev_float:
            obv_signal = 'buy'
        elif obv_float < obv_prev_float:
            obv_signal = 'sell'
        else:
            obv_signal = 'neutral'
        
        # Analyze VWAP - explicit float casting and comparison
        vwap_float = float(safe_get(latest_data, 'VWAP_D', 0.0))
        close_float = float(safe_get(latest_data, 'close', 0.0))
        if close_float < vwap_float:
            vwap_signal = 'buy'
        elif close_float > vwap_float:
            vwap_signal = 'sell'
        else:
            vwap_signal = 'neutral'
        
        # Analyze SMAs - explicit float casting and comparison
        sma20_float = float(safe_get(latest_data, 'sma_20', 0.0))
        sma50_float = float(safe_get(latest_data, 'sma_50', 0.0))
        if sma20_float > sma50_float:
            sma_signal = 'buy'
        elif sma20_float < sma50_float:
            sma_signal = 'sell'
        else:
            sma_signal = 'neutral'
        
        # Analyze price relative to Bollinger Bands - explicit float casting and comparison
        bb_upper_float = float(safe_get(latest_data, 'BBU_20_2.0', close_float*1.02))
        bb_lower_float = float(safe_get(latest_data, 'BBL_20_2.0', close_float*0.98))
        if close_float > bb_upper_float:
            bb_signal = 'sell'
        elif close_float < bb_lower_float:
            bb_signal = 'buy'
        else:
            bb_signal = 'neutral'
        
        # Store the float values to pass to our analysis dictionary
        macd_val = macd_float
        macd_signal_val = macd_signal_float
        obv_val = obv_float
        obv_prev = obv_prev_float
        vwap_val = vwap_float
        sma20 = sma20_float
        sma50 = sma50_float
        bb_upper = bb_upper_float
        bb_lower = bb_lower_float
        close_val = close_float
        
        # Compile all signals
        analysis = {
            'rsi': {
                'value': rsi_float,
                'signal': rsi_signal
            },
            'macd': {
                'value': macd_val,
                'signal_value': macd_signal_val,
                'signal': macd_signal
            },
            'obv': {
                'value': obv_val,
                'prev_value': obv_prev,
                'signal': obv_signal
            },
            'vwap': {
                'value': vwap_val,
                'signal': vwap_signal
            },
            'sma': {
                'sma20': sma20,
                'sma50': sma50,
                'signal': sma_signal
            },
            'bollinger_bands': {
                'upper': bb_upper,
                'lower': bb_lower,
                'signal': bb_signal
            },
            'close': close_val
        }
        
        return analysis
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error analyzing coin: {str(e)}")
        # Return neutral analysis if something goes wrong
        return {
            'rsi': {'value': 50, 'signal': 'neutral'},
            'macd': {'value': 0, 'signal_value': 0, 'signal': 'neutral'},
            'obv': {'value': 0, 'prev_value': 0, 'signal': 'neutral'},
            'vwap': {'value': 0, 'signal': 'neutral'},
            'sma': {'sma20': 0, 'sma50': 0, 'signal': 'neutral'},
            'bollinger_bands': {'upper': 0, 'lower': 0, 'signal': 'neutral'},
            'close': 0
        }

def should_enter_trade(analysis):
    """
    Determine if we should enter a trade based on technical analysis.
    Returns (should_trade, direction) where:
    - should_trade is a boolean
    - direction is 'long' or 'short'
    """
    try:
        # Count bullish and bearish signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Helper function to safely get string values
        def safe_get_signal(signals_dict, key, subkey='signal', default='neutral'):
            try:
                if key in signals_dict and isinstance(signals_dict[key], dict):
                    if subkey in signals_dict[key]:
                        signal = signals_dict[key][subkey]
                        # If signal is a pandas Series or numpy array, get first value
                        if isinstance(signal, (pd.Series, np.ndarray)):
                            if len(signal) > 0:
                                return str(signal.iloc[0] if hasattr(signal, 'iloc') else signal[0])
                            return default
                        return str(signal)  # Convert to string to ensure we have a scalar
                return default
            except Exception:
                return default
        
        # Check RSI - use safe_get_signal to ensure we're checking scalar strings
        rsi_signal = safe_get_signal(analysis, 'rsi')
        if rsi_signal == 'buy':
            bullish_signals += 1
        elif rsi_signal == 'sell':
            bearish_signals += 1
        
        # Check MACD
        macd_signal = safe_get_signal(analysis, 'macd')
        if macd_signal == 'buy':
            bullish_signals += 1
        elif macd_signal == 'sell':
            bearish_signals += 1
        
        # Check OBV
        obv_signal = safe_get_signal(analysis, 'obv')
        if obv_signal == 'buy':
            bullish_signals += 1
        elif obv_signal == 'sell':
            bearish_signals += 1
        
        # Check VWAP
        vwap_signal = safe_get_signal(analysis, 'vwap')
        if vwap_signal == 'buy':
            bullish_signals += 1
        elif vwap_signal == 'sell':
            bearish_signals += 1
        
        # Check SMA
        sma_signal = safe_get_signal(analysis, 'sma')
        if sma_signal == 'buy':
            bullish_signals += 1
        elif sma_signal == 'sell':
            bearish_signals += 1
        
        # Check Bollinger Bands
        bb_signal = safe_get_signal(analysis, 'bollinger_bands')
        if bb_signal == 'buy':
            bullish_signals += 1
        elif bb_signal == 'sell':
            bearish_signals += 1
        
        # Determine trading decision
        # For a stronger signal, require at least 4 indicators to agree
        if bullish_signals >= 4 and bullish_signals > bearish_signals:
            return True, 'long'
        elif bearish_signals >= 4 and bearish_signals > bullish_signals:
            return True, 'short'
        else:
            return False, None
    except Exception as e:
        import logging
        logger = logging.getLogger('trading_bot')
        logger.error(f"Error in should_enter_trade: {str(e)}")
        return False, None
