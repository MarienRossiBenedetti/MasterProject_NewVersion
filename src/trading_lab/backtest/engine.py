import pandas as pd

from trading_lab.backtest.portfolio import signal_to_pos, strat_rets, equity_curve
from trading_lab.data.loaders import import_yahoo
from trading_lab.strategies.strategies import sma_cross, rsi_cross
from trading_lab.indicators.indicators import add_sma, add_rsi

def engine_sma(ticker: str, start: str, end: str, fast_sma: int, slow_sma: int) -> pd.DataFrame:
    """
    Docstring
    Computes signal, pos, returns and equity curve in a single table 
    Used later in summary stats to backtest the strategy
    """
    # Import data and indicators (maybe outside)
    df = import_yahoo(ticker=ticker, start=start, end=end, raw=False)
    df = add_sma(df=df, length=fast_sma)
    df = add_sma(df=df, length=slow_sma)

    # Create signal column with strategy
    df = sma_cross(df=df, fast_ma=fast_sma, slow_ma=slow_sma)

    # Compute pos, rets and equity curve
    pos = signal_to_pos(df=df)
    rets = strat_rets(df=df, pos=pos)
    eq_curve = equity_curve(rets=rets)

    res = pd.DataFrame()
    res["pos"] = pos
    res["rets"] = (rets.where(rets.abs() > 1e-12, 0.0)).round(4)
    res["eq_curve"] = eq_curve.round(2)
    
    # Continue to print summary stats
    # engine doit etre générique pour toute les stratégies
    # pipeline import -> strat -> engine -> metrics

    return res

def engine_rsi(ticker: str, start: str, end:str, length:int, strips: list = [30, 70]) -> pd.DataFrame:
    """
    Docstring pour engine_rsi
    
    :param ticker: Description
    :type ticker: str
    :param start: Description
    :type start: str
    :param end: Description
    :type end: str
    :param length: Description
    :type length: int
    :return: Description
    :rtype: DataFrame
    """
    df = import_yahoo(ticker=ticker, start=start, end=end, raw=False)
    df = add_rsi(df=df, length=length)

    df = rsi_cross(df=df, length=length, strips=strips)

    # Compute pos, rets and equity curve
    pos = signal_to_pos(df=df)
    rets = strat_rets(df=df, pos=pos)
    eq_curve = equity_curve(rets=rets)

    res = pd.DataFrame()
    res["pos"] = pos
    res["rets"] = (rets.where(rets.abs() > 1e-12, 0.0)).round(4)
    res["eq_curve"] = eq_curve.round(2)

    return res