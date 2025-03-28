
import pandas as pd
import ta
import optuna
import ta.momentum
import ta.volatility

def indicadores_w_params(data: pd.DataFrame) -> pd.DataFrame:

    rsi = ta.momentum.RSIIndicator(data.Close, window=88)  # #Movimos manualmente los parametros de window, RSI_BUY, RSI_SELL
    bb = ta.volatility.BollingerBands(data.Close, window=15, window_dev=2)
    macd = ta.trend.MACD(data.Close)

    dataset = data.copy()
    dataset["RSI"] = rsi.rsi()
    dataset["BB"] = bb.bollinger_mavg()
    dataset["MACD"] = macd.macd_diff()

    dataset["RSI_BUY"] = dataset["RSI"] < 8
    dataset["RSI_SELL"] = dataset["RSI"] > 67

    dataset["BB_BUY"] = bb.bollinger_lband_indicator().astype(bool)
    dataset["BB_SELL"] = bb.bollinger_hband_indicator().astype(bool)

    dataset["MACD_BUY"] = dataset["MACD"] > 0
    dataset["MACD_SELL"] = dataset["MACD"] < 0

    return dataset.dropna()