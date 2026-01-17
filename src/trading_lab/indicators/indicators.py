import pandas as pd
import numpy as np

# Adding indicators to the stock prices table

def add_sma(df: pd.DataFrame, length: int) -> pd.DataFrame:
    """
    Docstring pour add_sma
    
    :param df: Description
    :type df: pd.DataFrame
    :param length: Description
    :type length: int
    :return: Description
    :rtype: DataFrame
    """
    df[f"sma_{length}"] = df["adj close"].rolling(window=length).mean()
    
    return df

def add_rsi(df: pd.DataFrame, length: int) -> pd.DataFrame:
    """
    Docstring pour add_rsi
    WILDER VERSION

    :param df: Description
    :type df: pd.DataFrame
    :param length: Description
    :type length: int
    :return: Description
    :rtype: DataFrame
    """
    delta_prices = df["adj close"].diff()

    gains = delta_prices.where(delta_prices > 0, 0)
    losses = (- delta_prices).where(delta_prices < 0, 0)

    # Wilder's smoothing
    avg_gain = gains.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df[f'RSI_{length}'] = rsi

    # Set first 'length' values to NaN (warmup period)
    df.loc[df.index[:length], f'RSI_{length}'] = np.nan

    return df

