import numpy as np
import pandas as pd
from sklearn import preprocessing

def get_means_covariances(prices: pd.DataFrame) -> tuple:
    """
    prices: dataframe com os dados de fechamento das ações ajustados. Cada coluna representa uma ação. Cada linha representa um dia.
    Esta função calcula as médias e covariâncias dos retornos diários das ações
    """
    means = prices.ffill().pct_change().dropna().values.mean(axis=0)
    covs = prices.ffill().pct_change().dropna().cov().values
    return means, covs

def get_initial_croms(num_croms, num_assets, seed=None):
    
    """
    num_croms: número de cromossomos a serem gerados aleatoriamente para a população inicial
    num_assets: número de ativos na carteira
    seed: semente para geração de números aleatórios
    Esta função gera a população inicial de cromossomos
    """

    if seed is not None:
        np.random.seed(seed)

    sorteios = np.random.randint(10, size=(num_croms, num_assets))
    return preprocessing.normalize(sorteios, norm='l1')

def get_return_wallet(crom, means):
    """
    crom: cromossomo
    means: médias dos retornos diários das ações
    Esta função calcula o retorno esperado da carteira (cromossomo)
    """
    return np.dot(crom, means)

def get_risk_wallet(crom, covs):
    """
    crom: cromossomo
    covs: matriz de covariâncias dos retornos diários das ações
    Esta função calcula o risco da carteira (cromossomo)
    """
    return np.dot(crom, np.dot(covs, crom.T))

def softmax(arr, base=1.05):
    """
    arr: array de números
    base: base da função softmax
    Esta função calcula a função softmax de um array de números para tornar os negativos em positivos pequenos e os positivos em positivos grandes
    """
    arr = base ** arr
    return arr

def get_fitnesses(wallets, means, covs):
    
    """
    wallets: carteiras (cromossomos) da população
    means: médias dos retornos diários das ações
    covs: matriz de covariâncias dos retornos diários das ações
    Esta função calcula os fitnesses (retorno esperado dividido pelo risco) de cada carteira (cromossomo) da população
    """
    
    returns = wallets.dot(means)

    risks = (wallets.dot(covs) * wallets).sum(axis=1)

    return returns / (risks + 1e-5)

def choose_parents(fitnesses_softmax, seed=None):
    
    """
    fitnesses_softmax: fitnesses (retorno esperado dividido pelo risco) de cada carteira (cromossomo) da população normalizados pela função softmax
    seed: semente para geração de números aleatórios
    Esta função escolhe os pais para o cruzamento de acordo com a roleta tendenciosa (biased roulette)
    """

    if seed is not None:
        np.random.seed(seed)

    fitnesses_percs = fitnesses_softmax / fitnesses_softmax.sum()
    return np.random.choice(range(len(fitnesses_percs)), size=2, p=fitnesses_percs, replace=False)

def crossover(cromossome_one, cromossome_two, n_sons=2, seed=None):
    
    """
    cromossome_one: cromossomo um para o cruzamento
    cromossome_two: cromossomo dois para o cruzamento
    n_sons: número de filhos a serem gerados
    seed: semente para geração de números aleatórios
    Esta função realiza o cruzamento de dois cromossomos para gerar filhos com características dos pais
    """

    if seed is not None:
        np.random.seed(seed)

    sons = []
    for _ in range(n_sons):
        alpha = np.random.uniform(0, 1)
        sons.append(alpha * cromossome_one + (1 - alpha) * cromossome_two)
    
    return sons

def mutation_one(son, seed=None):
    
    """
    son: filho para a mutação
    seed: semente para geração de números aleatórios
    Esta função realiza a mutação de um filho trocando dois genes aleatórios de posição
    """

    if seed is not None:
        np.random.seed(seed)

    genes_mutate = np.random.choice(range(son.size), size=2, replace=False)

    son = son.copy()

    son[genes_mutate[0]], son[genes_mutate[1]] = son[genes_mutate[1]], son[genes_mutate[0]]

    return [son]

def mutation_two(son, seed=None):
    
    """
    son: filho para a mutação
    seed: semente para geração de números aleatórios
    Esta função realiza a mutação de um filho somando dois genes aleatórios de posição e zerando um dos genes
    """

    if seed is not None:
        np.random.seed(seed)

    genes_mutate = np.random.choice(range(son.size), size=2, replace=False)

    son_one, son_two = son.copy(), son.copy()

    son_one[genes_mutate[0]] = son_one[genes_mutate].sum()
    son_one[genes_mutate[1]] = 0

    son_two[genes_mutate[0]] = 0
    son_two[genes_mutate[1]] = son_one[genes_mutate].sum()

    return [son_one, son_two]