from Data import cargar_datos
from Indicadores import indicadores
from backtesting_logic import ejecutar_backtesting
from Opt_Optuna import run_optimizacion
from Indicadores_w_params import indicadores_w_params
from backtesting_w_params import ejecutar_backtesting_w_params

def main():
    data = cargar_datos()
    data_indicadores = indicadores(data)

    backtesting_logic = ejecutar_backtesting(data_indicadores)

    #run_optimizacion(data_indicadores)

    data_indicadores_w_params = indicadores_w_params(data)
    backtesting_final = ejecutar_backtesting_w_params(data_indicadores_w_params)

if __name__ == "__main__":
    main()