import pandas as pd
import matplotlib.pyplot as plt

def ejecutar_backtesting(data_indicadores: pd.DataFrame) -> float:

    # --- Parámetros de estrategia ---
    capital = 1_000_000
    com = 0.125 / 100
    stop_loss = 0.05
    take_profit = 0.05
    n_shares = 1000

    portfolio_value = [capital]
    active_long_positions = None
    active_short_positions = None
    wins = 0
    losses = 0

    for _, row in data_indicadores.iterrows():
        # Cierre de posiciones
        if active_long_positions:
            if row.Close < active_long_positions["stop_loss"]:
                capital += row.Close * n_shares * (1 - com)
                active_long_positions = None
                losses += 1
            elif row.Close > active_long_positions["take_profit"]:
                capital += row.Close * n_shares * (1 - com)
                active_long_positions = None
                wins += 1

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

        # Apertura de posiciones
        if row.RSI_BUY and active_long_positions is None:
            cost = row.Close * n_shares * (1 + com)
            if capital > cost:
                capital -= cost
                active_long_positions = {
                    "datetime": row.Datetime,
                    "opened_at": row.Close,
                    "take_profit": row.Close * (1 + take_profit),
                    "stop_loss": row.Close * (1 - stop_loss)
                }

        if row.RSI_SELL and active_short_positions is None:
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

        # Calcular valor del portafolio
        long_value = row.Close * n_shares if active_long_positions else 0
        short_value = (active_short_positions["entry_amount"] - (row.Close * n_shares)) if active_short_positions else 0
        portfolio_value.append(capital + long_value + short_value)

    # --- Graficar evolución del portafolio y precio ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(portfolio_value, label="Portfolio Value")
    ax.set_title("Evolución del Portafolio vs Precio del Activo")
    ax.set_ylabel("Valor del Portafolio")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(data_indicadores["Close"].values, color="C1", alpha=0.5, label="Precio Close")
    ax2.set_ylabel("Precio del Activo")

    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    return portfolio_value[-1]


