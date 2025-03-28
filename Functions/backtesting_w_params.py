import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ejecutar_backtesting_w_params(data_indicadores: pd.DataFrame) -> dict:
    # --- Parámetros de estrategia ---
    capital = 1_000_000
    com = 0.125 / 100
    stop_loss = 0.15286820506904056  #Movimos manualmente los parametros de stop_loss, take_profit y n_shares
    take_profit = 0.06761982474536135
    n_shares = 3500

    portfolio_value = [capital]
    active_long_positions = None
    active_short_positions = None
    wins = 0
    losses = 0

    for _, row in data_indicadores.iterrows():
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

        # --- Señales combinadas ---
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

    # --- Calcular métricas ---
    portfolio = np.array(portfolio_value)
    returns = np.diff(portfolio) / portfolio[:-1]

    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    downside_returns = returns[returns < 0]
    sortino = np.mean(returns) / np.std(downside_returns) if len(downside_returns) > 0 else 0

    # Calmar Ratio con drawdown relativo
    running_max = np.maximum.accumulate(portfolio)
    drawdowns = (running_max - portfolio) / running_max  # relativo (%)
    max_drawdown = np.max(drawdowns)
    calmar = np.mean(returns) / max_drawdown if max_drawdown > 0 else 0

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    # --- Escalar métricas ---
    scaling_factor = np.sqrt(19656)
    sharpe *= scaling_factor
    sortino *= scaling_factor
    calmar *= scaling_factor

    # --- Mostrar métricas ---
    print("Métricas de desempeño")
    print(f"Final Portfolio Value: ${portfolio[-1]:,.2f}")
    print(f"Sharpe Ratio (anualizado): {sharpe:.3f}")
    print(f"Sortino Ratio (anualizado): {sortino:.3f}")
    print(f"Calmar Ratio (anualizado): {calmar:.3f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Wins: {wins}, Losses: {losses}, Total Trades: {total_trades}")

    # --- Gráfico del portafolio ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(portfolio, label="Portfolio Value")
    ax.set_title("Evolución del Portafolio vs Precio del Activo")
    ax.set_ylabel("Valor del Portafolio")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(data_indicadores["Close"].values, color="C1", alpha=0.5, label="Precio Close")
    ax2.set_ylabel("Precio del Activo")

    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # --- Devolver métricas como diccionario ---
    return {
        "Final Value": portfolio[-1],
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Win Rate": win_rate
    }
