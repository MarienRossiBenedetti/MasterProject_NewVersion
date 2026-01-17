import pandas as pd

def clean_table(df):
    """
    Docstring
    """
    df = df[['adj close']]
    return df