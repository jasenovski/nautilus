from utilidades.gerais import (gerar_data, jungir_retornos, 
                               gerar_carteira_aleatoria)

from modelo.moneta import moneta_ag
from math import log2
from cotacoes.cotacoes import busca_cotacoes, formata_cotacoes
import pandas as pd

def moneta_backtestes(data_inicial_bt, data_final_bt, 
                      intervalo, cotacoes_anteriores, cotacoes_segurar, maiores_medias,
                      qtd_bebados, cotacoes, cotacoes_index, simbolo_index) -> dict:
    
    """
    Função que executa o backteste do Moneta para uma configuração de parâmetros

    Args:
    data_inicial_bt (datetime): data inicial do backteste
    data_final_bt (datetime): data final do backteste
    intervalo (str): intervalo das cotações (d = diário, w = semanal)
    cotacoes_anteriores (int): quantidade de cotações anteriores para cada rodada do moneta
    cotacoes_segurar (int): quantidade de cotações para segurar a carteira para cada rodada do moneta
    maiores_medias (int): quantidade de maiores médias móveis para considerar na carteira para cada rodada do moneta
    qtd_bebados (int): quantidade de carteiras aleatórias para comparar com o Moneta
    cotacoes (pd.DataFrame): DataFrame com as cotações das ações
    cotacoes_index (pd.DataFrame): DataFrame com as cotações do índice
    simbolo_index (str): símbolo do índice a ser usado para comparar com o Moneta

    Returns:
    dict: dicionário com os resultados do backteste
    """
    
    resultados_moneta = []
    resultados_index = []
    resultados_bebados = []

    # data_rodar_moneta começa com a data inicial do backteste  
    data_rodar_moneta = data_inicial_bt

    todas_acoes = cotacoes.columns
    
    while data_rodar_moneta < data_final_bt:

        # data_inicial_moneta é a data inicial dos dados que serão usados para rodar o Moneta
        # até a data_rodar_moneta
        data_inicial_moneta = gerar_data(data_rodar_moneta, 
                                         cotacoes_anteriores, 
                                         intervalo, 
                                         "anterior")

        # data_final_testar_carteira é a data final dos dados que serão usados para testar a carteira
        data_final_testar_carteira = gerar_data(data_rodar_moneta, 
                                                cotacoes_segurar, 
                                                intervalo, 
                                                "posterior")

        # cotacoes_rodar_moneta é o DataFrame com as cotações que serão usadas para rodar o Moneta
        cotacoes_rodar_moneta = cotacoes.loc[data_inicial_moneta:data_rodar_moneta].copy()

        # formata as cotações para o intervalo e quantidade de maiores médias móveis
        variacoes_rodar_moneta = formata_cotacoes(cotacoes=cotacoes_rodar_moneta, 
                                                  maiores_medias=maiores_medias)
        
        print(variacoes_rodar_moneta)
        
        # resgata as ações presentes no DataFrame de variações
        acoes = variacoes_rodar_moneta.columns

        # roda o Moneta para otimizar a carteira
        carteira = moneta_ag(variacoes=variacoes_rodar_moneta)

        # resgata o retorno esperado da carteira e retira a função exponencial com o logaritmo
        retorno_esperado = log2(carteira.loc["Retornos"])

        # resgata apenas as ações presentes na carteira com seus percentuais
        # (sem as informações de retorno esperado, risco e fitness)
        carteira = carteira.loc[acoes]

        # resgata as cotações que serão usadas para testar a carteira
        # se a data final do backteste for menor que a data final para testar a carteira
        # então a data final para testar a carteira é a data final do backteste
        cotacoes_testar_carteira = cotacoes.loc \
            [data_rodar_moneta:min(data_final_bt, data_final_testar_carteira), acoes].copy()
        
        # formata as cotações para o intervalo e quantidade de maiores médias móveis
        variacoes_testar_carteira = formata_cotacoes(cotacoes=cotacoes_testar_carteira, 
                                                    maiores_medias=0)
        
        # resgata os retornos da carteira para cada dia dentro do DataFrame de cotações/variações
        retornos_moneta = variacoes_testar_carteira.dot(carteira)

        # resgata os retornos do índice para cada dia dentro do DataFrame de cotações do índice
        # se a data final do backteste for menor que a data final para testar a carteira
        # então a data final para testar a carteira é a data final do backteste
        cotacoes_index_testar = cotacoes_index.loc[data_rodar_moneta:min(data_final_bt, 
                                                                         data_final_testar_carteira)]

        # formata as cotações do índice para o intervalo
        # maiores médias móveis = 0 para não considerar nenhuma média móvel
        variacoes_index_testar = formata_cotacoes(cotacoes=pd.DataFrame(cotacoes_index_testar),
                                                maiores_medias=0)
        
        # resgata os retornos do índice para cada dia dentro do DataFrame de cotações/variações do índice
        retornos_index = variacoes_index_testar[simbolo_index]
        
        # calcula o retorno esperado do Moneta para o período de teste
        retorno_esperado_periodo = \
                    (1 + retorno_esperado) ** \
                    (data_final_testar_carteira - data_rodar_moneta).days - 1
        
        # salva os resultados do Moneta
        resultados_moneta.append(
            {
                "data_inicio": data_rodar_moneta,
                "data_fim": data_final_testar_carteira,
                "carteira": carteira,
                "retornos": retornos_moneta,
                "retorno_esperado": retorno_esperado_periodo
            }
        )

        # salva os resultados do índice
        resultados_index.append(
            {
                "data_inicio": data_rodar_moneta,
                "data_fim": data_final_testar_carteira,
                "retornos": retornos_index
            }
        )

        # salva os resultados dos bebados
        bebados = []
        for _ in range(qtd_bebados):

            # gera uma carteira aleatória com todas as ações disponíveis no DataFrame de cotações
            # de entrada
            carteira_aleatoria = gerar_carteira_aleatoria(acoes=todas_acoes, seed=None)
            acoes_aleatorias = carteira_aleatoria.index

            # resgata as cotações que serão usadas para testar as carteiras aleatórias
            cotacoes_testar_bebado = \
                cotacoes.loc[data_rodar_moneta:min(data_final_bt, data_final_testar_carteira), 
                             acoes_aleatorias].copy()
            
            # formata as cotações para o intervalo
            variacoes_testar_bebado = formata_cotacoes(cotacoes=cotacoes_testar_bebado,
                                                    maiores_medias=0)

            # computa os retornos da carteira aleatória para cada dia dentro do 
            # DataFrame de cotações/variações 
            retornos_bebado = variacoes_testar_bebado.dot(carteira_aleatoria)
            
            # salva os resultados do bebado
            dados_bebado = {"data_inicio": data_rodar_moneta,
                            "data_fim": data_final_testar_carteira,
                            "carteira": carteira_aleatoria,
                            "retornos": retornos_bebado}
            
            # adiciona os resultados do bebado na lista de resultados dos bebados
            bebados.append(dados_bebado)

        # adiciona os resultados dos bebados na lista de resultados dos bebados
        # para cada rodada do Moneta
        resultados_bebados.append(bebados)

        # atualiza a data para rodar o Moneta
        data_rodar_moneta = gerar_data(data_final_testar_carteira, 1, 
                                        intervalo, "posterior")
        
        print(f"Rodando Backteste do Moneta: {data_rodar_moneta}")
    
    # após todas as iterações (carteiras) do Moneta, calcula os retornos acumulados
    # no período inteiro de backteste do moneta
    retornos_jungidos_moneta = jungir_retornos(resultados_moneta, data_inicial_bt)
    resultados_acumulados_moneta = (retornos_jungidos_moneta + 1).cumprod()

    # após todas as iterações (carteiras) do Moneta, calcula os retornos acumulados
    # no período inteiro de backteste do índice
    retornos_jungidos_index = jungir_retornos(resultados_index, data_inicial_bt)
    resultados_acumulados_index = (retornos_jungidos_index + 1).cumprod()

    # após todas as iterações (carteiras) do Moneta, calcula os retornos acumulados
    # no período inteiro de backteste dos bebados
    resultados_acumulados_bebados = []
    for indice_bebado in range(qtd_bebados):
        # para cada bebado, resgata os retornos para cada rodada do Moneta
        retornos_bebado = [retornos[indice_bebado] 
                            for retornos in resultados_bebados]
        
        # junta os retornos de cada bebado para encontrar o retorno acumulado
        # do período inteiro de backteste
        retornos_jungidos_bebado = jungir_retornos(retornos_bebado, data_inicial_bt)

        # calcula os retornos acumulados do período inteiro de backteste
        resultados_acumulados_bebado = (retornos_jungidos_bebado + 1).cumprod()

        # adiciona os resultados acumulados do bebado na lista de resultados acumulados dos bebados
        resultados_acumulados_bebados.append(resultados_acumulados_bebado)

    # retorna um dicionário com os resultados acumulados, as variações diárias
    # do moneta e do índice e os dados gerais de cada rodada do Moneta

    return {
        "acumulados": {"moneta": resultados_acumulados_moneta,
                       "index": resultados_acumulados_index,
                       "bebados": resultados_acumulados_bebados},
        "variacoes": {"moneta": retornos_jungidos_moneta,
                      "index": retornos_jungidos_index},
        "dados": [{"moneta": rm, "index": ri} 
                    for rm, ri in zip(resultados_moneta, resultados_index)]
    }


# esta função vai parar no arquivo modelo/backtestes.py
def rodar_backtestes(acoes_selecionadas,
                    data_inicial_bt, data_final_bt, 
                    intervalo, cotacoes_anteriores, 
                    cotacoes_segurar, maiores_medias, qtd_bebados,
                    simbolo_index) -> dict:
    
    """
    Função que executa as preparações necessárias para rodar os backtestes do Moneta

    Args:
    acoes_selecionadas (list): lista com os símbolos das ações selecionadas
    data_inicial_bt (datetime): data inicial do backteste
    data_final_bt (datetime): data final do backteste
    intervalo (str): intervalo das cotações (d = diário, w = semanal)
    cotacoes_anteriores (int): quantidade de cotações anteriores para cada rodada do moneta
    cotacoes_segurar (int): quantidade de cotações para segurar a carteira para cada rodada do moneta
    maiores_medias (int): quantidade de maiores médias móveis para considerar na carteira para cada rodada do moneta
    qtd_bebados (int): quantidade de carteiras aleatórias para comparar com o Moneta
    simbolo_index (str): símbolo do índice a ser usado para comparar com o Moneta
    """
    
    # encontra a menor data para buscar as cotações que serão usadas em todos os backtestes
    # necessários
    data_minima = gerar_data(data_inicial_bt, cotacoes_anteriores, intervalo, "anterior")

    # encontra a maior data para buscar as cotações que serão usadas em todos os backtestes
    # necessários
    data_maxima = gerar_data(data_final_bt, cotacoes_segurar, intervalo, "posterior")

    # busca as cotações das ações selecionadas
    cotacoes = busca_cotacoes(simbolos=acoes_selecionadas,
                            intervalo=intervalo,
                            data_inicio=data_minima.strftime("%Y-%m-%d"),
                            data_fim=data_maxima.strftime("%Y-%m-%d"))
    
    # qualquer ação que não tenha dados retornados será removida do DataFrame de cotações
    cotacoes.dropna(axis=0, inplace=True)

    # busca as cotações do índice
    cotacoes_index = busca_cotacoes(simbolos=[simbolo_index],
                                    intervalo=intervalo,
                                    data_inicio=data_minima.strftime("%Y-%m-%d"),
                                    data_fim=data_maxima.strftime("%Y-%m-%d"))
    
    # qualquer dado dentro das cotações do índice que não tenha sido retornado será removido
    cotacoes_index.dropna(axis=0, inplace=True)

    # resgata as datas que estão presentes em ambos os DataFrames de cotações
    datas_comuns = cotacoes.index.intersection(cotacoes_index.index)

    # filtra os DataFrames de cotações das ações e do índice para ter apenas as datas
    # que estão presentes em ambos os DataFrames
    cotacoes = cotacoes.loc[datas_comuns]
    cotacoes_index = cotacoes_index.loc[datas_comuns]

    # chama a função que executa os backtestes do Moneta
    resultados_backtestes = moneta_backtestes(data_inicial_bt, data_final_bt, 
                                              intervalo, cotacoes_anteriores, cotacoes_segurar, 
                                              maiores_medias, qtd_bebados, cotacoes, cotacoes_index,
                                              simbolo_index=simbolo_index)

    return resultados_backtestes