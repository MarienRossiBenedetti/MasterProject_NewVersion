import pandas as pd
import numpy as np

# ------------ Return based metrics ------------ 

def annualized_ret(rets: pd.Series, periods_per_year: int = 252) -> float:
    """
    Docstring
    """
    return ((1 + rets).prod()) ** (periods_per_year / len(rets)) - 1

def annualized_vol(rets: pd.Series, periods_per_year: int = 252) -> float:
    """
    Docstring
    """
    return rets.std(ddof=0) * np.sqrt(periods_per_year)

def sharpe_ratio(rets: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Docstring
    """
    excess_mean = rets.mean() - (rf / periods_per_year)
    ann_vol = rets.std(ddof=0)
    return (excess_mean / ann_vol) * np.sqrt(periods_per_year)

# ------------ Equity based metrics ------------ 

def max_drawdown(eq_curve : pd.Series) -> float:
    """
    Docstring
    """
    eq_drdwn = eq_curve / eq_curve.cummax() - 1
    return - eq_drdwn.min()

# ------------ Trade quality ------------ 

def win_rate(rets: pd.Series) -> float:
    """
    Docstring
    """
    nbre_win = (rets > 0).sum()
    nbre_los = (rets < 0).sum()
    return nbre_win / (nbre_win + nbre_los)

def profit_factor(rets: pd.Series) -> float:
    """
    Docstring
    """
    gains = rets[rets > 0].sum()
    losses = - rets[rets < 0].sum()
    return gains / losses

# ------------- Summary statistics ------------- 

def summary_stats(rets: pd.Series, rf: int = 0.00, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Docstring
    """
    stats = {
        "Annualized Return" : annualized_ret(rets=rets, periods_per_year=periods_per_year),
        "Annualized Volatility" : annualized_vol(rets=rets, periods_per_year=periods_per_year),
        "Sharpe Ratio" : sharpe_ratio(rets=rets, rf=rf, periods_per_year=periods_per_year),
        "Max Drawdown" : max_drawdown((1 + rets).cumprod()),
        "Win Rate" : win_rate(rets=rets),
        "Profit Factor" : profit_factor(rets=rets),
    }
    
    return pd.DataFrame(stats, index=['Strategy']).T.round(4)
