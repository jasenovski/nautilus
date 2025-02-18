import pandas as pd
import numpy as np
from ag.ag import (gerar_cromossomos_base, roda_do_acaso, crossover,
                   mutacao_um, mutacao_dois, gerar_nova_geracao)

def moneta_ag(variacoes: pd.DataFrame, 
              qtd_iteracoes = 10, qtd_epocas = 40, qtd_croms_populacao_geral = 40):
        
    """
    Função que executa o algoritmo genético para otimização de carteiras do moneta

    Args:
    variacoes (pd.DataFrame): DataFrame com as variações periódicas das ações
    qtd_iteracoes (int): quantidade de iterações para cada época
    qtd_epocas (int): quantidade de épocas
    qtd_croms_populacao_geral (int): quantidade de cromossomos na população inicial

    Returns:
    pd.Series: cromossomo com a melhor carteira otimizada
    """

    # resgata as ações presentes no DataFrame de variações 
    acoes = list(variacoes.columns)

    # calcula a média e a matriz de covariância das variações
    medias = variacoes.mean(axis=0)
    matriz_covariancia = variacoes.cov()

    # gera os cromossomos iniciais da população
    cromossomos = gerar_cromossomos_base(qtd_croms_populacao_geral, acoes, medias, 
                                         matriz_covariancia)

    for epoca in range(qtd_epocas):

        # print(epoca)

        # sorteia 6 cromossomos para a roda do acaso
        indices_cromossomos_sorteados = \
            np.random.choice(cromossomos.index, size=6, replace=False)

        # resgata os cromossomos sorteados da população
        cromossomos_sorteados = cromossomos.loc[indices_cromossomos_sorteados]

        for iteracao in range(qtd_iteracoes):

            # print(iteracao)

            # RODA DO ACASO -------------------------------------
            # retorna os cromossomos pai e mãe sorteados
            cromossomo_pai, cromossomo_mae = roda_do_acaso(cromossomos_sorteados)
            
            # RODA DO ACASO -------------------------------------

            # CROSSOVER -----------------------------------------
            # retorna os cromossomos filhos
            cromossomo_filho_um = crossover(acoes, cromossomo_pai, cromossomo_mae)
            cromossomo_filho_dois = crossover(acoes, cromossomo_pai, cromossomo_mae)

            # CROSSOVER -----------------------------------------

            # MUTAÇÃO DO TIPO 1 ---------------------------------
            # retorna os cromossomos mutantes do tipo um
            mutante_um = mutacao_um(acoes, cromossomo_filho_um)
            mutante_dois = mutacao_um(acoes, cromossomo_filho_dois)
            # MUTAÇÃO DO TIPO 1 ---------------------------------

            # MUTAÇÃO DO TIPO 2 ---------------------------------
            # retorna os cromossomos mutantes do tipo dois
            mutante_tres, mutante_quatro = mutacao_dois(acoes, cromossomo_filho_um)
            mutante_cinco, mutante_seis = mutacao_dois(acoes, cromossomo_filho_dois)
            # MUTAÇÃO DO TIPO 2 --------------------------------

            # GERAÇÃO DA NOVA GERAÇÃO --------------------------
            # retorna a nova geração de cromossomos filhos/mutantes
            df_nova_geracao = gerar_nova_geracao(acoes, medias, matriz_covariancia, 
                                                 cromossomo_filho_um, cromossomo_filho_dois, 
                                                 mutante_um, mutante_dois, mutante_tres, 
                                                 mutante_quatro, mutante_cinco, mutante_seis)
            # GERAÇÃO DA NOVA GERAÇÃO --------------------------

            # recupera o indice do cromossomo com o pior fitness entre os 'pais'
            nome_cromossomo_ruim = cromossomos_sorteados["Fitnesses"].idxmin()

            # recupera o indice do cromossomo com o melhor fitness entre os 'filhos/mutantes'
            nome_cromossomo_bom = df_nova_geracao["Fitnesses"].idxmax()
            
            # recupera o fitness do pior pai e do melhor filho
            fitness_pior_pai = cromossomos_sorteados.loc[nome_cromossomo_ruim].loc["Fitnesses"]
            fitness_melhor_filho = df_nova_geracao.loc[nome_cromossomo_bom].loc["Fitnesses"]

            # se o fitness do melhor filho for maior que o fitness do pior pai
            # então o cromossomo do pior pai é substituído pelo cromossomo do melhor filho
            if fitness_melhor_filho > fitness_pior_pai:
                cromossomos_sorteados.loc[nome_cromossomo_ruim] = \
                    df_nova_geracao.loc[nome_cromossomo_bom].values
        
        # atualiza a população com os cromossomos sorteados iterados/melhorados
        cromossomos.loc[indices_cromossomos_sorteados] = \
            cromossomos_sorteados.values

    # recupera o indice do cromossomo com o melhor fitness
    indice_melhor_cromossomo = cromossomos["Fitnesses"].idxmax()

    # recupera o cromossomo com o melhor fitness após todas as épocas/iterações
    melhor_cromossomo = cromossomos.loc[indice_melhor_cromossomo]

    return melhor_cromossomo