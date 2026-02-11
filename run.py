import pandas as pd
from datetime import timedelta
from funcoes.utils import buscar_cotacoes, ajustar_prices
from funcoes.nautilus import nautilus
from funcoes.backtestes import backtestes_nautilus

def run_nautilus(stocks_selections, qtd_periodos, country, period, top_averages, epochs, times_run, total_croms, n_croms, base_softmax, seed, 
                 perc_max_nan, exportar_carteira):
    
    multiplicador = 1 if period == "d" else 7
    
    hoje = pd.Timestamp.today()
    data_inicio = hoje - pd.Timedelta(days=qtd_periodos * multiplicador)

    df_prices = buscar_cotacoes(start_date=data_inicio, end_date=hoje.strftime("%Y-%m-%d"), tickers_list=stocks_selections, country=country)
    df_prices = ajustar_prices(df_prices, period, perc_corte=perc_max_nan)

    nautilus(df_prices=df_prices, mm=top_averages, epochs=epochs, times_run=times_run, 
            total_croms=total_croms, n_croms=n_croms, base_softmax=base_softmax, seed=seed, exportar_carteira=exportar_carteira)

def run_backtestes(stocks_selections, data_iniciar_bt, data_terminar_bt, country, period,
                   index_id, periodos_anteriores, periodos_segurar, mm, epochs=10, times_run=30, total_croms=40, 
                   n_croms=6, base_softmax=1.10, seed=None, n_aleatorios=100, exportar_resultados=True):
    
    if period == "d":
        periods_previous = timedelta(days=periodos_anteriores)
    else:
        periods_previous = timedelta(weeks=periodos_anteriores)
    
    df_prices = buscar_cotacoes(start_date=data_iniciar_bt - periods_previous, end_date=data_terminar_bt, 
                                tickers_list=stocks_selections, country=country)
    df_prices = ajustar_prices(df_prices, period, perc_corte=0.03)

    if exportar_resultados:
        backtestes_nautilus(data_iniciar_bt=data_iniciar_bt, data_terminar_bt=data_terminar_bt, df_prices_ajustado=df_prices, index_id=index_id,
                            periodos_anteriores=periodos_anteriores, periodos_segurar=periodos_segurar, mm=mm, epochs=epochs, times_run=times_run, total_croms=total_croms, 
                            n_croms=n_croms, base_softmax=base_softmax, seed=seed, n_aleatorios=n_aleatorios, exportar_resultados=exportar_resultados)
    else:
        return backtestes_nautilus(data_iniciar_bt=data_iniciar_bt, data_terminar_bt=data_terminar_bt, df_prices_ajustado=df_prices, index_id=index_id,
                            periodos_anteriores=periodos_anteriores, periodos_segurar=periodos_segurar, mm=mm, epochs=epochs, times_run=times_run, total_croms=total_croms, 
                            n_croms=n_croms, base_softmax=base_softmax, seed=seed, n_aleatorios=n_aleatorios, exportar_resultados=exportar_resultados)
    
