import pickle as pkl
import streamlit as st
from variables import BR_STOCKS, US_STOCKS
from run import run_nautilus
import pandas as pd
import os
import numpy as np
import plotly.express as px
from funcoes.utils import data_vender
import datetime
from annotated_text import annotated_text

def pag_naut():

    st.title("Modelo Nautilus")
    st.markdown("Este modelo ajuda a navegar no mar de ações BR ou US e encontrar as melhores opções para a sua carteira de investimentos.")
    
    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    col1, col2 = st.sidebar.columns(2)

    country = col1.radio(
        label="Selecione o país:",
        options=["BR", "US"],
        index=0
    )

    period = col2.radio("Selecione o período:", ["Diário", "Semanal"], index=0)
    period = "d" if period == "Diário" else "w"

    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    check = st.sidebar.checkbox(
        label=f"Selecione se quiser rodar o modelo para todas as ações ({country})",
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

    # -----------------------------------------------------------------------------------------------------------------------------------------    
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    currency = "R$" if country == "BR" else "US$"
    investment = st.sidebar.text_input(
        label=f"Valor (base) do investimento na carteira ({currency}):",
        value="1000"
    )
    investment = float(investment.replace(",", "."))

    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    minimum_percentage = st.sidebar.text_input(
        label="Insira o valor mínimo para o percentual (%) aceitável de uma ação na carteira final:",
        value="5"
    )
    minimum_percentage = float(minimum_percentage.replace(",", ".")) / 100 if minimum_percentage != "" else 0.05
    
    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    qtd_cotacoes = st.sidebar.slider(
        label=f"Selecione o número de {'dias' if period == 'd' else 'semanas'} anteriores por ação:",
        min_value=7 if period == "d" else 4,
        max_value=200 if period == "d" else 50,
        value=30 if period == "d" else 10,
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
        value=15,
        step=1
    )

    # -----------------------------------------------------------------------------------------------------------------------------------------
    st.sidebar.divider()
    # -----------------------------------------------------------------------------------------------------------------------------------------

    solucao = st.sidebar.button(
        label="Gerar Carteira",
        on_click=run_nautilus,
        kwargs=
        {
        "stocks_selections": stocks_selections,
        "qtd_periodos": qtd_cotacoes,
        "country": country,
        "period": period,
        "top_averages": top_averages,
        "epochs": 10,
        "times_run": 30,
        "total_croms": 40,
        "n_croms": 6,
        "base_softmax": 1.10,
        "seed": None,
        "perc_max_nan": 0.03,
        "exportar_carteira": True,
        }
    )

    if solucao:

        # -----------------------------------------------------------------------------------------------------------------------------------------
        # Carregando os dados e tratando-os
        df_prices = pd.read_pickle(os.path.join("prices", "df_prices.pkl"))
        
        with open(os.path.join("resultados", "resultados.pkl"), "rb") as file:
            carteira, ret, risk = pkl.load(file)

        precos = df_prices[carteira.keys()].iloc[-1].values

        df_carteira = \
        pd.DataFrame(
            {
                "Ação": np.array(list(carteira.keys())),
                f"Preços ({currency})": precos,
                "Percentual %": np.array(list(carteira.values())),
                "Quantidade Ação": (np.array(list(carteira.values())) * investment / precos).round(0 if country == "BR" else 4),
                f"Valor Ação ({currency})": np.array(list(carteira.values())) * investment
            }
        )

        df_carteira = df_carteira[df_carteira["Percentual %"] >= minimum_percentage]
        df_carteira["Percentual %"] = (df_carteira["Percentual %"] * 100).round(2)
        # -----------------------------------------------------------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------------------------------------------------------
        # TABELA CARTEIRA
        st.subheader(
            body="Solução Obtida"
        )

        st.dataframe(
            data=df_carteira[["Ação", f"Preços ({currency})", "Percentual %", "Quantidade Ação", f"Valor Ação ({currency})"]]
        )

        dc = datetime.datetime.now()
        dv = data_vender(data_compra=dc, cotacoes_segurar=periodos_segurar, country=country)

        # st.warning(
        #     body=f"Data de venda da carteira: {dv.strftime('%d/%m/%Y')}",
        # )

        annotated_text(
            "Data de venda da carteira: ",
            (f"{dv.strftime('%d/%m/%Y')}", "⚠️", "#ff0")
        )
        # "#8ef" (azul claro)
        # "#ff0" (amarelo)
    
        
        # -----------------------------------------------------------------------------------------------------------------------------------------
        st.divider()
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # GRÁFICO
        fig = px.pie(df_carteira, values='Percentual %', names='Ação', title='Carteira Ótima')
        st.plotly_chart(fig)

        p = 22 if period == "d" else 4.28

        # CARDS
        col1, col2, col3 = st.columns(3)
        col1.metric("Valor Total", f"{currency} {df_carteira[f'Valor Ação ({currency})'].sum():_.2f}".replace(".", ",").replace("_", "."))
        col2.metric("Retorno a.m.", f"{((1 + ret) ** p - 1) * 100:.2f}%".replace(".", ","))
        col3.metric("Risco a.m.", f"{((1 + risk) ** p - 1) * 100:.2f}%".replace(".", ","))

        st.warning(
            body="O retorno mencionado acima é baseado em desempenho passado. Retorno passado não garante retorno futuro!!!",
            icon="⚠️"
        )

        # -----------------------------------------------------------------------------------------------------------------------------------------
        st.divider()
        # -----------------------------------------------------------------------------------------------------------------------------------------
        
        # TABELA COTAÇÕES
        st.subheader(f"Cotações (adj. close) para a geração da carteira ({currency}):")

        st.dataframe(
                data=df_prices
            )