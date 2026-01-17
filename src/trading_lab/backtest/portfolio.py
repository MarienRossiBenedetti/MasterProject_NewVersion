import pandas as pd

# Long only portfolios
# Usable on any dataframes with a 'adj close' and 'signal' columns (created by the strategy)

def signal_to_pos(df: pd.DataFrame) -> pd.Series:
    """
    Docstring
    """
    if "signal" not in df.columns:
        raise ValueError("⚠️ Missing 'Signal' Column")
    
    # Force a final sell order if ongoing buy order
    signal = df['signal'].values
    nz_signal = signal[signal != 0]
    if nz_signal.size == 0:
        raise ValueError("⚠️ No signal found - cannot proceed")
    if nz_signal[-1] == 1:                    # Buy order at the end
        df.loc[df.index[-1], "signal"] = -1   # Sell it at the final market price

    # Delete a would-be first sell signal
    if nz_signal[0] == -1:       
        first_idx_sell = df.index[df["signal"] == -1][0]
        df.loc[first_idx_sell, 'signal'] = 0

    # Convert signal into position
    pos = (df["signal"]
                .replace(0, pd.NA)
                .ffill()
                .replace(-1, 0)
                .fillna(0))
    
    return pos

def strat_rets(df: pd.DataFrame, pos: pd.Series) -> pd.Series:
    """
    Docstring
    """
    prices_rets = df['adj close'].pct_change().fillna(0)
    strat_rets = pos.shift(1) * prices_rets
    return strat_rets

def equity_curve(rets: pd.Series, init_wealth: float = 1000) -> pd.Series:
    """
    Docstring
    """
    return init_wealth * (1 + rets).cumprod()




