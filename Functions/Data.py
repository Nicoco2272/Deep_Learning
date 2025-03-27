import pandas as pd

def cargar_datos():
    data = pd.read_csv("aapl_5m_train.csv").dropna()
    return data
