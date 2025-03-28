from Data import cargar_datos
from Indicadores import indicadores
from backtesting_logic import ejecutar_backtesting
from Opt_Optuna import run_optimizacion

def main():
    data = cargar_datos()
    data_indicadores = indicadores(data)

    backtesting_logic = ejecutar_backtesting(data_indicadores)
    print(f"Valor final del portafolio: {backtesting_logic:,.2f}")

    #run_optimizacion(data_indicadores)

if __name__ == "__main__":
    main()