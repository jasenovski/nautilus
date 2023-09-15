import numpy as np
import pandas as pd
import streamlit as st
from variables import BR_STOCKS, US_STOCKS
from run import run_backtestes
import plotly.graph_objects as go
import pickle as pkl
from funcoes.performance_tracker import PerformanceTracker
import datetime as dtm

def pag_bts():

    col1, col2 = st.sidebar.columns(2)

    country = col1.radio(
        label="Selecione o país:",
        options=["BR", "US"],
        index=0
    )
    index_id = "BOVA11.SA" if country == "BR" else "^GSPC"

    period = col2.radio("Selecione o período:", ["Diário", "Semanal"], index=0)
    period = "d" if period == "Diário" else "w"

    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    check = st.sidebar.checkbox(
        label=f"Selecione se quiser rodar o backteste para todas as ações ({country})",
        value=False
    )

    stocks = sorted(BR_STOCKS if country == "BR" else US_STOCKS)
    if check is True:
        stocks_filtered = stocks[:]
    else:
        stocks_filtered = stocks[:5]

    stocks_selections = st.sidebar.multiselect(
        label="Ações Disponíveis:", 
        options=stocks,
        default=stocks_filtered
        )
    
    if index_id.replace(".SA", "") not in stocks_selections:
        stocks_selections.append(index_id)
    
    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    periodos_anteriores = st.sidebar.slider(
        label=f"Selecione o número de {'dias' if period == 'd' else 'semanas'} anteriores por ação:",
        min_value=7 if period == "d" else 4,
        max_value=200 if period == "d" else 50,
        value=90 if period == "d" else 20,
        step=1
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    periodos_segurar = st.sidebar.slider(
        label=f"Selecione o número de {'dias' if period == 'd' else 'semanas'} de cotação para segurar a carteira:",
        min_value=7 if period == "d" else 3,
        max_value=200 if period == "d" else 50,
        value=90 if period == "d" else 10,
        step=1
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    top_averages = st.sidebar.slider(
        label="Maiores médias:",
        min_value=0,
        max_value=20,
        value=20,
        step=1
    )
    
    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    qtd_aleatorios = st.sidebar.slider(
        label="Quantidade de carteiras aleatórias:",
        min_value=5,
        max_value=1000,
        value=100,
        step=1
    )

    
    st.title("Backtestes do Modelo Nautilus")
    st.markdown("Aqui você poderá realizar backtestes da configuração desejada do modelo Nautilus.")

    colunas = st.columns(2)
    data_inicial = pd.to_datetime(colunas[0].date_input(label="Data inicial", value=pd.to_datetime("2019-01-01"), min_value=pd.to_datetime("2015-01-01")))
    data_final = pd.to_datetime(colunas[1].date_input(label="Data final", value=pd.to_datetime(dtm.datetime.now().strftime("%Y-%m-%d")), min_value=data_inicial))

    solucao = st.sidebar.button(
        label="Gerar Backteste",
        on_click=run_backtestes,
        kwargs=
        {
        "stocks_selections": stocks_selections, 
        "country": country,
        "period": period,
        "data_iniciar_bt": data_inicial, 
        "data_terminar_bt": data_final, 
        "index_id": index_id, 
        "periodos_anteriores": periodos_anteriores, 
        "periodos_segurar": periodos_segurar, 
        "mm": top_averages, 
        "epochs": 100, 
        "times_run": 1000, 
        "total_croms": 40, 
        "n_croms": 6,
        "base_softmax": 1.10, 
        "seed": None, 
        "n_aleatorios": qtd_aleatorios, 
        "perc_max_nan": 0.03,
        "exportar_resultados": True
        }
    )

    if solucao:
        with open("resultados/resultados_backteste.pkl", "rb") as f:
            resultados = pkl.load(f)
        
        patrimonio_moneta = resultados[0][0]
        patrimonio_index = resultados[0][1]
        patrimonios_aleatorios = resultados[0][2]

        retornos_moneta = resultados[1][0]
        retornos_index = resultados[1][1]

        carteiras = resultados[2]

        tracker = PerformanceTracker(data_returns=retornos_moneta, market_returns=retornos_index - 1, annual_risk_free=0.02, period=period)
        sharpe_ratio = tracker.sharpe_ratio()
        beta = tracker.portfolio_beta()
        annualized_return = tracker.annualized_return()
        max_drawdown = tracker.max_drawdown()

        # resultados_gerais = resultados[2]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=patrimonio_moneta.index, y=patrimonio_moneta, name="Moneta", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=patrimonio_index.index, y=patrimonio_index, name=f"{index_id}", line=dict(color="red")))

        aleatorios = sorted(np.random.choice(range(len(patrimonios_aleatorios)), size=len(patrimonios_aleatorios) // 2, replace=False))
        for aleatorio in aleatorios:
            fig.add_trace(go.Scatter(x=patrimonios_aleatorios[aleatorio].index, y=patrimonios_aleatorios[aleatorio], name=f"Aleatorio {aleatorio}", line=dict(color="green"))) \
            .update_traces(visible="legendonly", selector=lambda t: not t.name in ["Moneta", f"{index_id}"])

        st.plotly_chart(fig)

        resultado_final_moneta = 100 * (patrimonio_moneta.iloc[-1] - 1)
        resultado_final_index = 100 * (patrimonio_index.iloc[-1] - 1)
        delta = resultado_final_moneta - resultado_final_index

        # CARDS
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Retorno Moneta", f"{resultado_final_moneta:.0f}%", delta=f"{delta:.2f}%")
        col2.metric("Retorno Anual", f"{annualized_return:.0f}%")
        col3.metric("Beta", f"{beta:.2f}")
        col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col5.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        st.subheader("Carteiras geradas no backteste")
        st.divider()
        for i, carteira in enumerate(carteiras):
            data_inicial = carteira["data_inicial"]
            data_final = carteira["data_final"]
            wallet = carteira["carteira"]
            retorno_esperado = carteira["retorno_esperado"]
            risco_esperado = carteira["risco_esperado"]

            df = (pd.DataFrame(wallet.values(), index=wallet.keys(), columns=["Percentuais %"]) * 100).round(2)
            df = df[df["Percentuais %"] > 1]

            

            st.write(f"Carteira {i + 1}")
            st.write(f"{data_inicial.strftime('%d/%m/%Y')} até {data_final.strftime('%d/%m/%Y')}")
            st.dataframe(df)

            col1 = st.columns(1)
            col1[0].metric("Retorno Esperado", f"{retorno_esperado * 100:.2f}%")
            st.divider()