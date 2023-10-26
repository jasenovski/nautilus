import numpy as np
from funcoes.nautilus import nautilus
from funcoes.utils import gerar_carteira_aleatoria
import pickle as pkl
import os
import pandas as pd
from stqdm import stqdm

def backtestes_nautilus(data_iniciar_bt, data_terminar_bt, df_prices_ajustado, index_id,
    periodos_anteriores, periodos_segurar, mm, epochs, times_run, total_croms, n_croms, 
    base_softmax, seed, n_aleatorios, exportar_resultados):
    
    prices_index = df_prices_ajustado[[index_id]]
    
    carteiras_moneta = []
    data_rodar_moneta = data_iniciar_bt

    LENGTH = len(df_prices_ajustado[df_prices_ajustado.index > data_rodar_moneta].index[::periodos_segurar])
    progress = stqdm(total=LENGTH, desc="Rodando backtestes", unit="rodada")# , ncols=100, mininterval=1.0)
    INDEX = 0
    while data_rodar_moneta < data_terminar_bt:

        df_prices_ajustado_moneta = df_prices_ajustado[df_prices_ajustado.index < data_rodar_moneta].iloc[-periodos_anteriores:]
        df_prices_ajustado_futuro = df_prices_ajustado[df_prices_ajustado.index > data_rodar_moneta].iloc[:periodos_segurar]

        if len(df_prices_ajustado_futuro) == 0:
            break

        carteira, ret, risk = nautilus(df_prices=df_prices_ajustado_moneta, mm=mm, epochs=epochs, 
                          times_run=times_run, total_croms=total_croms,
                          n_croms=n_croms, base_softmax=base_softmax, seed=seed)
        
        acoes_carteira = list(carteira.keys())
        percentuais_carteira = np.array(list(carteira.values())).reshape(-1, 1)
        retornos_moneta = df_prices_ajustado_futuro[acoes_carteira].pct_change().fillna(0).values.dot(percentuais_carteira).ravel()
        retornos_moneta_series = pd.Series(retornos_moneta, index=df_prices_ajustado_futuro.index)

        retornos_aleatorios = []
        for _ in range(n_aleatorios):
            carteira_aleatoria = gerar_carteira_aleatoria(acoes=df_prices_ajustado.columns, seed=seed)
            percentuais_carteira_aleatoria = np.array(list(carteira_aleatoria.values())).reshape(-1, 1)
            acoes_carteira_aleatoria = list(carteira_aleatoria.keys())
            # retornos = df_prices_ajustado_futuro[acoes_carteira_aleatoria].pct_change().dropna().values.dot(percentuais_carteira_aleatoria).ravel()
            retornos = df_prices_ajustado_futuro[acoes_carteira_aleatoria].pct_change().fillna(0).values.dot(percentuais_carteira_aleatoria).ravel()
            # retornos_series = pd.Series(retornos, index=df_prices_ajustado_futuro.index[1:])
            retornos_series = pd.Series(retornos, index=df_prices_ajustado_futuro.index)
            retornos_aleatorios.append(retornos_series)
        
        data_final_simulacao = df_prices_ajustado_futuro.index[-1]
        
        prices_index_filtrado = prices_index[prices_index.index > data_rodar_moneta].iloc[:periodos_segurar]
        retornos_index = prices_index_filtrado[[index_id]].pct_change().fillna(0).values.ravel()
        retornos_index_series = pd.Series(retornos_index, index=prices_index_filtrado.index)

        carteiras_moneta.append(
            {
                "data_inicial": data_rodar_moneta,
                "data_final": data_final_simulacao,
                "retornos_moneta": retornos_moneta_series,
                "retornos_index": retornos_index_series,
                "retornos_aleatorios": retornos_aleatorios,
                "carteira": carteira,
                "retorno_esperado": (1 + ret) ** (data_final_simulacao - data_rodar_moneta).days - 1,
                "risco_esperado": risk,
            }
        )

        data_rodar_moneta = data_final_simulacao
        INDEX += 1
        progress.update(1)
    
    resultados_moneta = pd.Series([0], index=[data_iniciar_bt])
    resultados_index = pd.Series([0], index=[data_iniciar_bt])
    for resultado in carteiras_moneta:
        resultados_moneta = pd.concat([resultados_moneta, resultado["retornos_moneta"]])
        resultados_index = pd.concat([resultados_index, resultado["retornos_index"]])

    
    carteiras_aleatorios = [pd.Series([0], index=[data_iniciar_bt]) for _ in range(n_aleatorios)]
    for i, carteira_aleatorio in enumerate(carteiras_aleatorios):
        for retornos in carteiras_moneta:
            retornos_aleatorio = retornos["retornos_aleatorios"][i]
            carteira_aleatorio = pd.concat([carteira_aleatorio, retornos_aleatorio])
    
        carteiras_aleatorios[i] = carteira_aleatorio
    
    patrimonio_acumulado_moneta = (1 + resultados_moneta).cumprod()
    patrimonio_acumulado_index = (1 + resultados_index).cumprod()

    patrimonios_acumulados_aleatorios = []
    for retornos_aleatorios in carteiras_aleatorios:
        patrimonios_acumulados_aleatorios.append((1 + retornos_aleatorios).cumprod())
    
    if exportar_resultados:
        with open(os.path.join("resultados", "resultados_backteste.pkl"), "wb") as f:
            pkl.dump([[patrimonio_acumulado_moneta, patrimonio_acumulado_index, patrimonios_acumulados_aleatorios], [resultados_moneta, resultados_index], carteiras_moneta], f)
    else:
        return [patrimonio_acumulado_moneta, patrimonio_acumulado_index, patrimonios_acumulados_aleatorios], [resultados_moneta, resultados_index], carteiras_moneta

