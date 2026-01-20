import pandas as pd
import numpy as np
from trading_lab.indicators.indicators import add_sma, add_rsi

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Docstring pour create_features
    
    :param df: Description
    """
    df_features = df.copy()
    df_features = df.rename(columns={'Adj Close':'adj close'})

    df_features = add_sma(df_features, 20)
    df_features = add_sma(df_features, 50)
    df_features = add_sma(df_features, 200)

    df_features['price_to_sma_20'] = df_features['adj close'] / df_features['sma_20'] - 1
    df_features['price_to_sma_50'] = df_features['adj close'] / df_features['sma_50'] - 1

    df_features = add_rsi(df_features, 14)
    df_features = add_rsi(df_features, 28)

    df_features['rets_1d'] = df_features['adj close'].pct_change()
    df_features['rets_5d'] = df_features['adj close'].pct_change(5)
    df_features['rets_10d'] = df_features['adj close'].pct_change(10)

    df_features['vol_20d'] = df_features['rets_1d'].rolling(20).std()
    
    df_features['volume_ratio_20'] = df_features['Volume'] / df_features["Volume"].rolling(20).mean()

    df_features = df_features.drop(columns=["Close", "High", "Low", "Open", "Volume", "adj close"])

    return df_features

def create_target(df: pd.DataFrame, horizon: int=5, buy_trsh: float=.02, sell_trsh:float=-.02) -> pd.Series:
    """
    Docstring pour create_target
    
    :param df: Description
    :type df: pd.DataFrame
    :param horizon: Description
    :type horizon: int
    :param buy_trsh: Description
    :type buy_trsh: float
    :param sell_trsh: Description
    :type sell_trsh: float
    :return: Description
    :rtype: Series[Any]
    """
    df_target = df.copy()

    df_target = df_target.rename(columns={'Adj Close':'adj close'})

    future_rets = df_target['adj close'].pct_change(horizon).shift(-horizon)

    target = pd.Series(0, index=df_target.index, name='target')
    target[future_rets > buy_trsh] = 1
    target[future_rets < sell_trsh] = -1

    return target

def train_test_split(X: pd.DataFrame, y:pd.Series, train_ratio: float=0.6, val_ratio: float=0.2) -> dict:
    """
    Docstring pour train_test_split
    
    :param X: Description
    :type X: pd.DataFrame
    :param y: Description
    :type y: pd.Series
    :param train_ratio: Description
    :type train_ratio: float
    :param val_ratio: Description
    :type val_ratio: float
    :return: Description
    :rtype: dict
    """
    n = X.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    split_data = {
        'X_train' : X.iloc[:train_end],
        'y_train' : y.iloc[:train_end],
        'X_val': X.iloc[train_end:val_end],
        'y_val': y.iloc[train_end:val_end],
        'X_test': X.iloc[val_end:],
        'y_test': y.iloc[val_end:]
    }

    return split_data

def process_data(df: pd.DataFrame, horizon: int=5, buy_trsh: float=.02, sell_trsh:float=-.02) -> tuple[pd.DataFrame, pd.Series]:
    """
    Docstring pour process_data
    
    :param df: Description
    :type df: pd.DataFrame
    :param horizon: Description
    :type horizon: int
    :param buy_trsh: Description
    :type buy_trsh: float
    :param sell_trsh: Description
    :type sell_trsh: float
    :return: Description
    :rtype: tuple[DataFrame, Series[Any]]
    """
    df_features = create_features(df)
    target = create_target(df=df, horizon=horizon, buy_trsh=buy_trsh, sell_trsh=sell_trsh)
    df_features['target'] = target
 
    df_clean = df_features.dropna()

    X = df_clean.drop(columns=['target'])
    X.columns.name = None

    y = df_clean['target']

    return X, y