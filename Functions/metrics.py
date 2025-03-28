import numpy as np

def calculo_metricas(returns, portfolio_value):
    returns = np.array(returns)

    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    downside_returns = returns[returns < 0]
    sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if np.std(downside_returns) > 0 else 0

    max_portfolio_value = np.max(portfolio_value)
    calmar_ratio = (portfolio_value[-1] - portfolio_value[0]) / (max_portfolio_value - portfolio_value[0]) if (max_portfolio_value - portfolio_value[0]) > 0 else 0

    return sharpe_ratio, sortino_ratio, calmar_ratio

def imprimir_metricas(sharpe, sortino, calmar, wins, losses):
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print("\n--- Métricas de la Estrategia ---")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Calmar Ratio: {calmar:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")