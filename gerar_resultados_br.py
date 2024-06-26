import pandas as pd
from itertools import product
from run import run_backtestes
from funcoes import utils
from funcoes.performance_tracker import PerformanceTracker
import pickle as pkl

country = "br"
if country == "br":
    index_id = "BOVA11.SA"
    from variables import BR_STOCKS as acoes
    acoes.remove("HAPV3")
    acoes.remove("NTCO3")
    acoes.remove("RRRP3")
    acoes.remove("SIMH3")
elif country == "us":
    index_id = "^GSPC"
    from variables import US_STOCKS as acoes

epochs = 10
times_run = 10
total_croms = 40
n_croms = 6
base_softmax = 1.10
seed = None
n_aleatorios = 100

# comecos = ["2018-03-15", "2018-08-10", "2019-01-28", "2019-06-04", "2020-02-17", "2020-08-15", "2021-03-21", "2021-08-11"]
comecos = ["2014-05-05", "2014-09-10", "2015-03-15", "2015-08-10", "2016-01-28", "2016-06-04", "2017-02-17", "2017-08-15", "2018-03-21", "2018-08-11"]

# finais = ["2022-04-20", "2022-08-02", "2023-02-11", "2023-06-22", "2023-09-01"]
finais = ["2019-04-20", "2019-08-02", "2020-02-11", "2020-06-22", "2020-09-01", "2020-12-30"]

segurar = [(30, 4), (90, 12), (200, 25)]
maiores_medias = [-1, 10, 20]
cotacoes_anteriores = [(30, 10), (90, 30), (200, 100)]
periodos = ["d", "w"]
k_final = len(comecos) * len(finais) * len(segurar) * len(maiores_medias) * len(cotacoes_anteriores) * len(periodos)

nome = "resultados.pkl"

resultados = []

for k, (period, se, mm, ca, data_inicio, data_final) in list(enumerate(product(periodos, segurar, maiores_medias, cotacoes_anteriores, comecos, finais))):

    print(f"{k + 1:05}/{k_final:05}")
    s = se[0] if period == "d" else se[1]
    a = ca[0] if period == "d" else ca[1]

    data_iniciar_bt = pd.to_datetime(data_inicio)
    data_terminar_bt = pd.to_datetime(data_final)
    
    patrimonios, retornos, _ = run_backtestes(stocks_selections=acoes, country=country, period=period,
                                                data_iniciar_bt=data_iniciar_bt, data_terminar_bt=data_terminar_bt, index_id=index_id, periodos_anteriores=a, 
                                                periodos_segurar=s, mm=mm, epochs=epochs, times_run=times_run, total_croms=total_croms, n_croms=n_croms,
                                                base_softmax=base_softmax, seed=seed, n_aleatorios=n_aleatorios, perc_max_nan=0.10, exportar_resultados=False)

    quartis_moneta, quartis_index, _ = utils.gerar_quartis(*patrimonios, size=5)

    tracker = PerformanceTracker(data_returns=retornos[0], market_returns=retornos[1], period=period)
    max_drawdown_moneta = tracker.max_drawdown()
    beta_moneta = tracker.portfolio_beta()
    sharpe_moneta = tracker.sharpe_ratio()


    resultados.append({"data inicio": data_inicio, "data final": data_final, 
                        "cotacoes segurar": s, "maiores medias": mm, "cotacoes anteriores": a, "period": period,
                        "q1 moneta": quartis_moneta[0], "q2 moneta": quartis_moneta[1], "q3 moneta": quartis_moneta[2], 
                        f"q1 {index_id.replace('.SA', '').lower()}": quartis_index[0], f"q2 {index_id.replace('.SA', '').lower()}": quartis_index[1], f"q3 {index_id.replace('.SA', '').lower()}": quartis_index[2], 
                        "patrimonio final moneta": patrimonios[0].iloc[-1], f"patrimonio final {index_id.replace('.SA', '').lower()}": patrimonios[1].iloc[-1],
                        "sharpe": sharpe_moneta, "beta": beta_moneta, "max drawdown": max_drawdown_moneta})

    if k % 10 == 0:
        with open(nome, "wb") as f:
            pkl.dump([k, resultados], f)

with open(nome, "wb") as f:
    pkl.dump([k, resultados], f)