import pandas as pd
import numpy as np

from trading_lab.indicators.indicators import add_sma, add_rsi

# -------------------------------------------------
# Strategies create a column 'signal' stacked to the 
# original dataframe

###################################################
### -------------- MOVING AVERAGES ------------ ###
###################################################

def sma_cross(df: pd.DataFrame, fast_ma: int, slow_ma: int) -> pd.DataFrame:
    """
    Docstring pour sma_cross
    
    :param df: Description
    :type df: pd.DataFrame
    :param fast_ma: Description
    :type fast_ma: int
    :param slow_ma: Description
    :type slow_ma: int
    :return: Description
    :rtype: DataFrame
    """
    if fast_ma >= slow_ma:
        raise ValueError("⚠️ fast must be < than slow")

    df = add_sma(df, fast_ma) # Add fast ma
    df = add_sma(df, slow_ma) # Add slow ma
    df.dropna(inplace=True)

    # Rename columns for convenience
    sma_fast = f'sma_{fast_ma}'
    sma_slow = f'sma_{slow_ma}'

    # Relative positions of smas
    fast_above = df[sma_fast] > df[sma_slow]
    prev_fast_above = fast_above.shift(1)

    # Cross events
    bullish_cross = fast_above & (prev_fast_above == False)
    bearish_cross = (fast_above == False) & prev_fast_above

    # Write signals
    df["signal"] = 0
    df.loc[bullish_cross, "signal"] = 1
    df.loc[bearish_cross, "signal"] = -1

    return df   

###################################################
### ------------------ MOMENTUM --------------- ###
###################################################

def rsi_cross(df: pd.DataFrame, length: int, strips: list = [30, 70]) -> pd.DataFrame:
    
    ### VERIFIER LA LOGIQUE DE LA STRATEGIE
    ### WHEN TO BUY AND WHEN TO SELL
    
    """
    Docstring pour rsi_cross
    
    :param df: Description
    :type df: pd.DataFrame
    :param length: Description
    :type length: int
    :return: Description
    :rtype: DataFrame
    """
    if length <= 0:
        raise ValueError("⚠️ length must be positive")
    df = add_rsi(df=df, length=length)
    df.dropna(inplace=True)

    lower_strip, upper_strip = strips

    rsi_above = df[f'RSI_{length}'] > upper_strip
    prev_rsi_above = rsi_above.shift(1)
    rsi_below = df[f'RSI_{length}'] < lower_strip
    prev_rsi_below = rsi_below.shift(1)

    bullish_cross = rsi_above & (prev_rsi_above == False)
    bearish_cross = rsi_below & (prev_rsi_below == False)

    df["raw_signal"] = 0
    df.loc[bullish_cross, "raw_signal"] = 1
    df.loc[bearish_cross, "raw_signal"] = -1

    df["position"] = df["raw_signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

    df["signal"] = df["position"].diff().fillna(0).astype(int)

    df["signal"] = df["signal"].clip(-1, 1)

    df.drop(["raw_signal", "position"], axis=1, inplace=True)

    df.attrs["RSI_strips"] = strips

    return df