
import pandas as pd
import ta
import optuna
import ta.momentum
import ta.volatility

def indicadores(data: pd.DataFrame) -> pd.DataFrame:

    rsi = ta.momentum.RSIIndicator(data.Close, window=20)  # Ajustable por optimización
    bb = ta.volatility.BollingerBands(data.Close, window=15, window_dev=2)

    dataset = data.copy()
    dataset["RSI"] = rsi.rsi()
    dataset["BB"] = bb.bollinger_mavg()

    dataset["RSI_BUY"] = dataset["RSI"] < 25
    dataset["RSI_SELL"] = dataset["RSI"] > 75

    dataset["BB_BUY"] = bb.bollinger_lband_indicator().astype(bool)
    dataset["BB_SELL"] = bb.bollinger_hband_indicator().astype(bool)

    return dataset.dropna()
