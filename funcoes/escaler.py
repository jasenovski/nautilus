import numpy as np

def initial_tuples(n, p_max, step, result=()):
    if n == 0:
        yield result
    else:
        for i in range(0, p_max + 1, step):
            yield from initial_tuples(n - 1, p_max - i, step, result + (i, ))

def generate_all_asset_combinations(n_ativos, step):
    result = list(initial_tuples(n_ativos - 1, 100, step))
    final_result = []
    for row in result:
        new_row = row + (100 - sum(row), )
        final_result.append(new_row)
    return final_result

def escaler(df_prices, step=5):
    combinations = generate_all_asset_combinations(n_ativos=len(df_prices.columns), step=step)

    arr = np.array(combinations) / 100
    
    retornos_diarios = df_prices.pct_change().dropna().values

    medias_retornos = retornos_diarios.mean(axis=0).reshape(-1, 1)

    matriz_covs = np.cov(retornos_diarios.T)

    retornos = arr.dot(medias_retornos)
    riscos = (arr.dot(matriz_covs) * arr).sum(axis=1).reshape(-1, 1)
    fos = retornos / (riscos + 1e-10)

    ind_max_fo = fos.argsort(axis=0)[::-1][0][0]

    wallet = arr[ind_max_fo]
    ret = retornos[ind_max_fo][0]
    risco = riscos[ind_max_fo][0]

    wallet = {acao: val for acao, val in zip(df_prices.columns, wallet)}

    return wallet, ret, risco

