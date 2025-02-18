import pandas as pd
import numpy as np
from ces.ces import ces_retornos, ces_riscos, ces_fitnesses
from sklearn import preprocessing


def gerar_nova_geracao(acoes: list, medias: pd.Series, matriz_covariancia: pd.DataFrame, 
                       cromossomo_filho_um: pd.Series, 
                       cromossomo_filho_dois: pd.Series, mutante_um: pd.Series, 
                       mutante_dois: pd.Series, mutante_tres: pd.Series, 
                       mutante_quatro: pd.Series, mutante_cinco: pd.Series, 
                       mutante_seis: pd.Series) -> pd.DataFrame:
    
    """
    Função que gera a nova geração de cromossomos a partir dos cromossomos
    filhos e mutantes gerados na função de cruzamento e mutação.

    Args:
    - acoes: lista com os nomes dos ativos
    - medias: médias dos retornos dos ativos
    - matriz_covariancia: matriz de covariância dos retornos dos ativos
    - cromossomo_filho_um: cromossomo filho gerado no cruzamento
    - cromossomo_filho_dois: cromossomo filho gerado no cruzamento
    - mutante_um: cromossomo mutante gerado na mutação
    - mutante_dois: cromossomo mutante gerado na mutação
    - mutante_tres: cromossomo mutante gerado na mutação
    - mutante_quatro: cromossomo mutante gerado na mutação
    - mutante_cinco: cromossomo mutante gerado na mutação
    - mutante_seis: cromossomo mutante gerado na mutação

    Returns:
    - df_nova_geracao: DataFrame com os cromossomos da nova
    """
    
    df_nova_geracao = pd.DataFrame(data=[cromossomo_filho_um, cromossomo_filho_dois,
                                                    mutante_um, mutante_dois,
                                                    mutante_tres, mutante_quatro,
                                                    mutante_cinco, mutante_seis])
            
    df_nova_geracao["Retornos"] = \
                ces_retornos(carteiras=df_nova_geracao, medias=medias)

    df_nova_geracao["Riscos"] = ces_riscos(
                carteiras=df_nova_geracao.loc[:, acoes], 
                matriz_covariancia=matriz_covariancia)

    df_nova_geracao["Fitnesses"] = ces_fitnesses(
                retornos=df_nova_geracao.loc[:, "Retornos"],
                riscos=df_nova_geracao.loc[:, "Riscos"]
            )
    
    return df_nova_geracao


def mutacao_dois(acoes: list, cromossomo_filho: pd.Series) -> pd.Series:
    """
    Função que realiza a mutação do tipo dois em um cromossomo filho.

    Args:
    - acoes: lista com os nomes dos ativos
    - cromossomo_filho: cromossomo filho gerado no cruzamento

    Returns:
    - mutante_a: cromossomo mutante gerado na mutação
    - mutante_b: outro cromossomo mutante gerado na mutação
    """

    genes_sorteados = np.random.choice(acoes, size=2, replace=False)
    soma_genes = cromossomo_filho.loc[genes_sorteados].sum()
    mutante_a = cromossomo_filho.copy()
    mutante_a.loc[genes_sorteados[0]] = soma_genes
    mutante_a.loc[genes_sorteados[1]] = 0

    mutante_b = cromossomo_filho.copy()
    mutante_b.loc[genes_sorteados[0]] = 0
    mutante_b.loc[genes_sorteados[1]] = soma_genes
    return mutante_a,mutante_b

def mutacao_um(acoes: list, cromossomo_filho: pd.Series) -> pd.Series:
    """
    Função que realiza a mutação do tipo um em um cromossomo filho.

    Args:
    - acoes: lista com os nomes dos ativos
    - cromossomo_filho: cromossomo filho gerado no cruzamento
    """

    genes_sorteados = np.random.choice(acoes, size=2, replace=False)
    mutante = cromossomo_filho.copy()
    mutante.loc[genes_sorteados] = \
                cromossomo_filho.loc[genes_sorteados].iloc[::-1].values
        
    return mutante

def crossover(acoes: list, cromossomo_pai: pd.Series, cromossomo_mae: pd.Series) -> pd.Series:

    """
    Função que realiza o cruzamento entre dois cromossomos.

    Args:
    - acoes: lista com os nomes dos ativos
    - cromossomo_pai: cromossomo pai
    - cromossomo_mae: cromossomo mãe

    Returns:
    - cromossomo_filho: cromossomo filho gerado no cruzamento
    """

    al = np.random.rand()
    parte_genes_pai = al * cromossomo_pai.loc[acoes]
    parte_genes_mae = (1 - al) * cromossomo_mae.loc[acoes]
    cromossomo_filho = parte_genes_mae + parte_genes_pai
    return cromossomo_filho

def roda_do_acaso(cromossomos_sorteados: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Função que realiza a roda do acaso para selecionar os cromossomos pais.

    Args:
    - cromossomos_sorteados: DataFrame com os cromossomos sorteados

    Returns:
    - cromossomo_pai: cromossomo pai
    - cromossomo_mae: cromossomo mãe
    """

    # gera um series com as percentagens relativas
    # 0.2, 0.45, 0.05, 0.10, 0.15, 0.05
    percentagens_relativas_fitnesses = \
                cromossomos_sorteados.loc[:, "Fitnesses"] / \
                    cromossomos_sorteados.loc[:, "Fitnesses"].sum()

    # gera um series com as percentagens acumuladas
    # 0.2, 0.65, 0.70, 0.80, 0.95, 1.00
    percentagens_acumuladas_fitnesses = \
                percentagens_relativas_fitnesses.cumsum()
                
    # esse comando gera um aleatorio de 0 até 1
    # ex. 0.68
    al = np.random.rand()

    # retorna a posição do cromossomo sorteado
    # no exemplo acima seria o cromossomo de índice
    # 2 (terceiro cromossomo)
    posicao_cromossomo_sorteado = \
                (al > percentagens_acumuladas_fitnesses).sum()

    # retorna o cromossomo pai
    cromossomo_pai = cromossomos_sorteados.iloc[posicao_cromossomo_sorteado]

    # o cromossomo mãe começa igual ao cromossomo pai
    cromossomo_mae = cromossomo_pai.copy()

    # enquanto o cromossomo mãe for igual ao cromossomo pai
    # sorteie novos aleatórios até que o cromossomo mãe seja
    # diferente do cromossomo pai
    while (cromossomo_mae == cromossomo_pai).all():
        al = np.random.rand()
        posicao_cromossomo_sorteado = \
                    (al > percentagens_acumuladas_fitnesses).sum()

        cromossomo_mae = cromossomos_sorteados.iloc[posicao_cromossomo_sorteado]
    
    # retorna o cromossomo pai e mãe
    return cromossomo_pai, cromossomo_mae

def gerar_cromossomos_base(qtd_croms_populacao_geral: int, 
                           acoes: list, medias: pd.Series, 
                           matriz_covariancia: pd.DataFrame) -> pd.DataFrame:
    
    """
    Função que gera os cromossomos base da população inicial.

    Args:
    - qtd_croms_populacao_geral: quantidade de cromossomos na população inicial
    - acoes: lista com os nomes dos ativos
    - medias: médias dos retornos dos ativos
    - matriz_covariancia: matriz de covariância dos retornos dos ativos

    Returns:
    - cromossomos: DataFrame com os cromossomos da população inicial
    """

    qtd_genes = len(acoes)

    carteiras = np.random.randint(low=0, high=10, 
                                size=(qtd_croms_populacao_geral, qtd_genes))
    cromossomos = preprocessing.normalize(carteiras, norm="l1", axis=1)
    cromossomos = pd.DataFrame(data=cromossomos, columns=acoes)

    cromossomos["Retornos"] = ces_retornos(cromossomos, medias)
    cromossomos["Riscos"] = ces_riscos(cromossomos.loc[:, acoes],
                                    matriz_covariancia)
    cromossomos["Fitnesses"] = ces_fitnesses(cromossomos.loc[:, "Retornos"],
                                            cromossomos.loc[:, "Riscos"])
                                            
    return cromossomos