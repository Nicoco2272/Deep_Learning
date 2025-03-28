import pandas as pd
import optuna
import ta

def objective_func(trial, data: pd.DataFrame) -> float:
    # Hiperparámetros a optimizar
    rsi_window = trial.suggest_int("rsi_window", 10, 100)
    rsi_lower = trial.suggest_int("rsi_lower", 5, 35)
    rsi_upper = trial.suggest_int("rsi_upper", 65, 95)
    stop_loss = trial.suggest_float("stop_loss", 0.01, 0.2)
    take_profit = trial.suggest_float("take_profit", 0.01, 0.2)
    n_shares = trial.suggest_categorical("n_shares", [1000, 2000, 3000, 3500, 4000])

    # Indicadores técnicos
    rsi = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    bb = ta.volatility.BollingerBands(data.Close, window=15, window_dev=2)
    macd = ta.trend.MACD(data.Close)

    dataset = data.copy()
    dataset["RSI"] = rsi.rsi()
    dataset["BB"] = bb.bollinger_mavg()
    dataset["MACD"] = macd.macd()
    dataset["MACD_SIGNAL"] = macd.macd_signal()

    # Señales RSI
    dataset["RSI_BUY"] = dataset["RSI"] < rsi_lower
    dataset["RSI_SELL"] = dataset["RSI"] > rsi_upper

    # Señales BB
    dataset["BB_BUY"] = bb.bollinger_lband_indicator().astype(bool)
    dataset["BB_SELL"] = bb.bollinger_hband_indicator().astype(bool)

    # Señales MACD
    dataset["MACD_BUY"] = macd.macd_diff() > 0
    dataset["MACD_SELL"] = macd.macd_diff() < 0

    dataset = dataset.dropna()

    # --- Estrategia de backtesting ---
    capital = 1_000_000
    com = 0.125 / 100
    portfolio_value = [capital]
    active_long_positions = None
    active_short_positions = None
    wins = 0
    losses = 0

    for _, row in dataset.iterrows():
        # Cierre de posiciones largas
        if active_long_positions:
            if row.Close < active_long_positions["stop_loss"]:
                capital += row.Close * n_shares * (1 - com)
                active_long_positions = None
                losses += 1
            elif row.Close > active_long_positions["take_profit"]:
                capital += row.Close * n_shares * (1 - com)
                active_long_positions = None
                wins += 1

        # Cierre de posiciones cortas
        if active_short_positions:
            exit_amount = row.Close * n_shares
            pnl = active_short_positions["entry_amount"] - exit_amount
            pnl -= exit_amount * com
            if row.Close > active_short_positions["stop_loss"]:
                capital += pnl
                active_short_positions = None
                losses += 1
            elif row.Close < active_short_positions["take_profit"]:
                capital += pnl
                active_short_positions = None
                wins += 1

        # Señales combinadas
        buy_signals = sum([row.RSI_BUY, row.BB_BUY, row.MACD_BUY])
        sell_signals = sum([row.RSI_SELL, row.BB_SELL, row.MACD_SELL])

        # Abrir posición larga (BUY)
        if buy_signals >= 2 and active_long_positions is None:
            cost = row.Close * n_shares * (1 + com)
            if capital > cost:
                capital -= cost
                active_long_positions = {
                    "datetime": row.Datetime,
                    "opened_at": row.Close,
                    "take_profit": row.Close * (1 + take_profit),
                    "stop_loss": row.Close * (1 - stop_loss)
                }

        # Abrir posición corta (SELL)
        if sell_signals >= 2 and active_short_positions is None:
            entry_amount = row.Close * n_shares
            commission_cost = entry_amount * com
            if capital > commission_cost:
                capital -= commission_cost
                active_short_positions = {
                    "datetime": row.Datetime,
                    "opened_at": row.Close,
                    "entry_amount": entry_amount,
                    "take_profit": row.Close * (1 - take_profit),
                    "stop_loss": row.Close * (1 + stop_loss)
                }

        # Valor actual del portafolio
        long_value = row.Close * n_shares if active_long_positions else 0
        short_value = (active_short_positions["entry_amount"] - (row.Close * n_shares)) if active_short_positions else 0
        portfolio_value.append(capital + long_value + short_value)

    return portfolio_value[-1]

def run_optimizacion(data: pd.DataFrame):

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_func(trial, data), n_trials=50)

    print(f" El mejor valor final encontrado fue de: ${study.best_value:,.2f}")
    print(" Para llegar a este valor final, los parámetros usados fueron los siguientes:")
    for k, v in study.best_params.items():
        print(f"   - {k}: {v}")

    return study.best_params