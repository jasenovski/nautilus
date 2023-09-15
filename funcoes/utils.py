import pandas_datareader.data as web
import yfinance as yf
import numpy as np
import pandas as pd
from variables import FERIADOS_US, FERIADOS_BR
import datetime

def buscar_cotacoes(start_date, end_date, tickers_list: list, country: str) -> tuple:

    """
    tickers_list: lista de tickers das ações
    dias_cotacoes: quantidade de dias para download dos dados de fechamento
    country: país de origem das ações
    period: período de cotações. 'd' para diário, 'w' para semanal, 'm' para mensal
    Esta função realiza a busca dos dados de fechamento das cotações para cada dia
    """

    # tickers_df = pd.read_csv(arquivo_txt, header=None)
    tickers = []
    if country.upper() == "US":
        for ticker in tickers_list:
            tickers.append(ticker)
    elif country.upper() == "BR":
        for ticker in tickers_list:
            tickers.append(ticker if ticker.endswith(".SA") else ticker + ".SA")  # ação BR deve terminar com '.SA'

    yf.pdr_override()
    cotations = web.get_data_yahoo(tickers, start=start_date, end=end_date , threads=1)['Adj Close']

    return cotations

def gerar_carteira_aleatoria(acoes, seed=None):

    """
    acoes: lista de ações
    seed: semente para geração de números aleatórios
    Esta função gera uma carteira aleatória com pesos aleatórios para cada ação
    """

    if seed:
        np.random.seed(seed)
    
    n_acoes = np.random.randint(1, len(acoes) + 1)

    acoes_escolhidas = np.random.choice(acoes, size=n_acoes, replace=False)
    sorteio = np.random.randint(1, 11, size=n_acoes)
    percentuais = sorteio / sorteio.sum()
    return {acao: percentual for acao, percentual in zip(acoes_escolhidas, percentuais)}

def ajustar_prices(df, period="d", perc_corte=0.05):

    """
    df: dataframe com os dados de fechamento das ações. Cada coluna representa uma ação. Cada linha representa um dia.
    period: período de cotações. 'd' para diário, 'w' para semanal
    perc_corte: percentual de corte para excluir ações com muitos valores faltantes
    Esta função exclui ações com muitos valores faltantes e exclui linhas com valores faltantes
    """

    soma_na_por_acao = df.isna().sum()
    acoes_excluir = soma_na_por_acao[soma_na_por_acao > perc_corte * len(df)].index
    for acao_excluir in acoes_excluir:
        df.drop(acao_excluir, axis=1, inplace=True)
    
    df.dropna(axis=0, inplace=True)

    df = df.iloc[::1 if period == "d" else 5]

    return df

def gerar_quartis(patrimonio_acum_moneta, patrimonio_acum_index, patrimonios_aleatorios, size=3):

    """
    patrimonio_acum_moneta: patrimônio acumulado da carteira Moneta
    patrimonio_acum_index: patrimônio acumulado do índice BOVA11
    patrimonios_aleatorios: patrimônio acumulado das carteiras aleatórias
    size: quantidade de carteiras aleatórias para retornar os quartis
    Esta função retorna os quartis para o patrimônio acumulado da carteira Moneta, do índice BOVA11 e das carteiras aleatórias
    """

    n_aleatorios = len(patrimonios_aleatorios)

    arr = np.zeros(shape=(len(patrimonios_aleatorios) + 2, len(patrimonio_acum_moneta)), dtype=np.float64)
    arr[0, :] = patrimonio_acum_moneta
    arr[1, :] = patrimonio_acum_index

    for i, cumprod_aleatorios in enumerate(patrimonios_aleatorios, 2):
        arr[i:, :] = cumprod_aleatorios

    asort = np.argsort(arr[:, 1:], axis=0)

    quartis_moneta = np.quantile(n_aleatorios + 2 - np.where(asort == 0)[0], q=[0.25, 0.5, 0.75])
    quartis_bova = np.quantile(n_aleatorios + 2 - np.where(asort == 1)[0], q=[0.25, 0.5, 0.75])

    quartis_aleatorios = []
    for i in np.random.choice(range(n_aleatorios), size=size, replace=False):
        quartis_aleatorios.append(np.quantile(n_aleatorios + 2 - np.where(asort == i + 2)[0], q=[0.10, 0.25, 0.5, 0.75, 0.90]))

    return quartis_moneta, quartis_bova, quartis_aleatorios

# def fo(media_quartis, media_retorno, media_ganha_index, a=1, b=1, c=1):
#     return a * media_quartis + b * media_retorno + c * media_ganha_index

def fo(media_quartis, media_patrimonio, media_vs_bova, media_sharpe, media_beta, media_max_drawdown,
       a, b, c, d, e, f):
    
    return a * media_quartis + b * media_patrimonio + c * media_vs_bova + d * media_sharpe + e * media_beta + f * media_max_drawdown

def gerar_ranking(df_resultados, qtd_aleatorios, cols=["cotacoes segurar", "maiores medias", "cotacoes anteriores", "period"], country="br"):
    combinacoes = df_resultados[cols].drop_duplicates().reset_index(drop=True)

    index_id = "bova" if country == "br" else "^gspc"

    ranking = []
    for i, comb in combinacoes.iterrows():
        df_comb = df_resultados[(df_resultados["cotacoes segurar"] == comb["cotacoes segurar"]) &
                    (df_resultados["maiores medias"] == comb["maiores medias"]) &
                    (df_resultados["cotacoes anteriores"] == comb["cotacoes anteriores"]) &
                    (df_resultados["period"] == comb["period"])]
        
        pontuacao_quartis = (qtd_aleatorios - (df_comb["q1 moneta"].mean() + df_comb["q2 moneta"].mean() + df_comb["q3 moneta"].mean()) / 3) / qtd_aleatorios

        pontuacao_patrimonio = (df_comb["patrimonio final moneta"] / df_resultados["patrimonio final moneta"].max()).mean()

        pontuacao_vs_index = ((1.1 ** (df_comb["patrimonio final moneta"] - df_comb[f"patrimonio final {index_id}"])) / (1.1 ** (df_resultados["patrimonio final moneta"] - df_resultados[f"patrimonio final {index_id}"])).max()).mean()

        pontuacao_sharpe = (1.1 ** df_comb["sharpe"] / (1.1 ** df_resultados["sharpe"]).max()).mean()

        pontuacao_beta = 1 - (df_comb["beta"] / df_resultados["beta"].max()).mean()

        pontuacao_max_drawdown = (-1 * df_comb["max drawdown"] / (-1 * df_resultados["max drawdown"]).max()).mean()

        ob = fo(pontuacao_quartis, pontuacao_patrimonio, pontuacao_vs_index, pontuacao_sharpe, pontuacao_beta, pontuacao_max_drawdown,
                1, 1, 1, 1, 1, 1)
        

        ranking.append({
            "indice_comb": i,
            "cotacoes segurar": comb["cotacoes segurar"],
            "maiores medias": comb["maiores medias"],
            "cotacoes anteriores": comb["cotacoes anteriores"],
            "period": comb["period"],
            "media quartis": pontuacao_quartis,
            "media patrimonio": pontuacao_patrimonio,
            "media vs index": pontuacao_vs_index,
            "media sharpe": pontuacao_sharpe,
            "media beta": pontuacao_beta,
            "media max drawdown": pontuacao_max_drawdown,
            "fo": ob
        })
    df_final = pd.DataFrame(ranking).sort_values(by="fo", ascending=False).reset_index(drop=True)
    df_final["fo_ajustada"] = 10 * df_final["fo"] / df_final["fo"].max()
    return df_final

def data_vender(data_compra, cotacoes_segurar, country):

    feriados = FERIADOS_BR if country.lower() == "br" else FERIADOS_US

    list_total_days = [(data_compra + datetime.timedelta(x + 1)).strftime("%Y-%m-%d") for x in range(cotacoes_segurar * 2)]
    business_days = [day for day in list_total_days if datetime.datetime.strptime(day, "%Y-%m-%d").date().weekday() < 5 and day not in feriados]

    dv = business_days[cotacoes_segurar - 1]
    dv = datetime.datetime.strptime(dv, "%Y-%m-%d").date()

    return dv
