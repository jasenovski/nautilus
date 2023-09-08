import os
import pickle as pkl
import numpy as np
import pandas as pd
from funcoes.ag import (get_means_covariances, get_initial_croms, get_return_wallet,
                         get_risk_wallet, softmax, get_fitnesses, choose_parents, 
                         crossover, mutation_one, mutation_two)

def nautilus(df_prices: pd.DataFrame, mm: int = 10, epochs: int = 30, times_run: int = 100, 
           total_croms: int = 40, n_croms: int = 6, base_softmax=1.05, seed=None, exportar_carteira: bool = False) -> dict:
        
    """
    df_prices: dataframe com os dados de fechamento das ações ajustados. Cada coluna representa uma ação. Cada linha representa um dia.
    mm: número de ações com maiores médias de variação diária a serem usadas no algoritmo
    epochs: número de épocas a serem rodadas no algoritmo
    times_run: número de vezes que o algoritmo será rodado
    total_croms: número total de cromossomos a serem gerados aleatoriamente para a população inicial
    n_croms: número de cromossomos a serem usados em cada época
    base_softmax: base da função softmax
    seed: semente para geração de números aleatórios
    investiment: valor a ser investido na carteira
    Esta função roda o algoritmo genético para gerar a carteira ótima de ações. Retorna um dicionário com a carteira ótima, o retorno esperado e o risco esperado
    """

    if mm > 0:
        greatest_means = df_prices.ffill().pct_change().dropna().mean().argsort()[::-1]
        stocks_mm = greatest_means[:mm].index
        df_prices = df_prices[stocks_mm]

    tickers = df_prices.columns

    means, covs = get_means_covariances(df_prices)

    wallets_geral = get_initial_croms(num_croms=total_croms, num_assets=len(tickers), seed=seed)
    for _ in range(times_run):
                
        croms_selected = np.random.choice(range(total_croms), size=n_croms, replace=False)
        wallets = wallets_geral[croms_selected]

        for _ in range(epochs):
            fitnesses = get_fitnesses(wallets, means, covs)
            fitnesses_softmax = softmax(fitnesses, base=base_softmax)

            parents = choose_parents(fitnesses_softmax, seed=seed)
            regular_sons = crossover(wallets[parents[0]], wallets[parents[1]], n_sons=2, seed=seed)

            son_mutation_one = mutation_one(regular_sons[0], seed=seed)
            son_mutation_two = mutation_one(regular_sons[1], seed=seed)
            sons_mutation_three = mutation_two(regular_sons[0], seed=seed)
            sons_mutation_four = mutation_two(regular_sons[1], seed=seed)
            newgen = np.concatenate([regular_sons, son_mutation_one, son_mutation_two, sons_mutation_three, sons_mutation_four])

            fitnesses_newgen = get_fitnesses(newgen, means, covs)
            fitnesses_newgen_softmax = softmax(fitnesses_newgen, base=base_softmax)

            arg_max_sons = fitnesses_newgen_softmax.argsort()[::-1]
            arg_min_parents = fitnesses_softmax.argsort()

            wallets[arg_min_parents[0]] = newgen[arg_max_sons[0]]
            wallets[arg_min_parents[1]] = newgen[arg_max_sons[1]]
        
        wallets_geral[croms_selected] = wallets
    
    fitnesses_geral = get_fitnesses(wallets_geral, means, covs)
    fitnesses_geral_softmax = softmax(fitnesses_geral, base=base_softmax)

    arg_max_geral = fitnesses_geral_softmax.argsort()[::-1]
    wallet_vencedora = wallets_geral[arg_max_geral[0]]
    
    if exportar_carteira:
        resultados = [dict(zip(tickers, wallet_vencedora)), get_return_wallet(wallet_vencedora, means), get_risk_wallet(wallet_vencedora, covs)]
        with open(os.path.join("resultados", "resultados.pkl"), "wb") as f:
            pkl.dump(resultados, f)
        
        with open(os.path.join("prices", "df_prices.pkl"), "wb") as f:
            pkl.dump(df_prices, f)

    else:
        return [dict(zip(tickers, wallet_vencedora)), get_return_wallet(wallet_vencedora, means), get_risk_wallet(wallet_vencedora, covs)]