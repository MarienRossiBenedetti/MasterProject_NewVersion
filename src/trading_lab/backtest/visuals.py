import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visuals_sma_strat(df: pd.DataFrame, eq_curve: pd.Series, buy_sell_signals: bool = True) -> None:
    """
    Docstring pour visuals_sma_strat
    
    :param df: as returned by sma_cross
    :param eq_curve: Description
    """
    # Also plot equity curve?????????
    df['adj close'].plot(c="#201D1D", linewidth=1, label=df.attrs['name'])
    df.iloc[:, 1].plot(c="#BD5810", linestyle='--')
    df.iloc[:, 2].plot(c="#285D13", linestyle='--')
    if buy_sell_signals:
        buys = df['signal'] == 1
        sells = df["signal"] == -1
        plt.scatter(df.index[buys], df["adj close"][buys], marker='^', c="#0C12CA", s=80, label='Buy', zorder=5)
        plt.scatter(df.index[sells], df["adj close"][sells], marker='v', c="#B60505", s=80, label='Sell', zorder=5)
    plt.ylabel("Price")
    plt.grid(True, linestyle='--')
    plt.title(f"SMA crossover strategy for {df.attrs['name']}", fontweight='bold')
    plt.legend()
    plt.show()

def visuals_rsi_strat(df: pd.DataFrame, eq_curve: pd.Series, buy_sell_signals: bool = True) -> None:
    """
    Docstring pour visuals_rsi_strat
        
    :param df: Description
    :type df: pd.DataFrame
    :param eq_curve: Description
    :type eq_curve: pd.Series
    :param buy_sell_signals: Description
    :type buy_sell_signals: bool
    """
    # Also plot equity curve?????????
    lower_strip, upper_strip = df.attrs.get('RSI_strips')
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axs
    # Stock price
    ax1.plot(df.index, df['adj close'], c="#201D1D", linewidth=1, label=df.attrs['name'])
    # RSI indicator and strips
    ax2.plot(df.index, df.iloc[:, 1], c="#610A7E", alpha=0.8, label=df.columns[1])
    ax2.axhline(lower_strip, linestyle='--', c="#D5DF11")
    ax2.axhline(upper_strip, linestyle='--', c="#D5DF11")
    # Buy and Sell signals
    if buy_sell_signals:
        buys = df['signal'] == 1
        sells = df["signal"] == -1
        ax1.scatter(df.index[buys], df["adj close"][buys], marker='^', c="#0C12CA", s=80, label='Buy', zorder=5)
        ax1.scatter(df.index[sells], df["adj close"][sells], marker='v', c="#B60505", s=80, label='Sell', zorder=5)
    ax1.set_ylabel("Price")
    plt.grid(True, linestyle='--')

    plt.suptitle(f"RSI crossover strategy for {df.attrs['name']}", fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.show()

        
