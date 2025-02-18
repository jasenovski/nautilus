import streamlit as st
from datetime import date, datetime
from modelo.backtestes import rodar_backtestes
import pandas as pd
import plotly.graph_objects as go
from utilidades.performance_tracker import PerformanceTracker

def pagina_backtestes(simbolos, paises, intervalos):

    # cria o título da página
    st.title(body="Modelo Backtestes")
    st.write("Realiza backtestes para o moneta")

    # cria um espaço para duas colunas para inserir widgets
    colunas = st.columns(2)

    # na primeira coluna, cria um widget 'date_input' para selecionar a data inicial
    # do backteste
    data_inicial = colunas[0].date_input(label="data inicial", 
                                         value=date(2019, 1, 1))

    # na segunda coluna, cria um widget 'date_input' para selecionar a data final
    # do backteste
    data_final = colunas[1].date_input(label="data final",
                                       min_value=data_inicial,
                                       value=datetime.now().date())
    
    st.divider()

    # ---------------------------------------------------
    # cria duas colunas no sidebar para inserir os widgets
    colunas = st.sidebar.columns(2)

    # na primeira coluna, cria um widget 'radio' para selecionar o país
    pais = colunas[0].radio(label="Selecione uma bolsa de ações", options=paises, index=0)

    # na segunda coluna, cria um widget 'radio' para selecionar o intervalo
    intervalo = colunas[1].radio(label="Selecione o intervalo", options=intervalos, index=0)
    st.sidebar.divider()
    # ---------------------------------------------------

    # ---------------------------------------------------
    # cria um widget 'checkbox' para selecionar todas as ações do país selecionado
    flag_acoes = st.sidebar.checkbox(label=f"Selecionar todas as ações do país ({pais})")
    if flag_acoes:
        bolsa_acoes = simbolos[paises[pais]][1:]
    else:
        # se o widget 'checkbox' não estiver marcado, selecionar apenas as 5 primeiras ações
        bolsa_acoes = simbolos[paises[pais]][1:6]
    
    # cria um widget 'multiselect' para apresentar as ações possíveis de serem selecionadas
    # para backteste
    acoes_selecionadas = st.sidebar.multiselect(label="Selecione as ações para rodar o modelo",
                                                options=simbolos[paises[pais]][1:],
                                                default=bolsa_acoes)

    st.sidebar.divider()
    # ---------------------------------------------------

    # ---------------------------------------------------
    # cria um widget 'slider' para selecionar a quantidade de cotações anteriores
    qtd_cotacoes_anteriores = st.sidebar.slider(
        label=f"Selecione a quantidade de " \
        f"{'dias' if intervalo == 'Diário' else 'semanas'} anteriores",
        min_value=5,
        value=200,
        max_value=500,
        step=1
        )
    st.sidebar.divider()
    # ---------------------------------------------------

    # ---------------------------------------------------
    # cria um widget 'slider' para selecionar a quantidade de cotações a segurar
    qtd_cotacoes_segurar = st.sidebar.slider(
        label=f"Selecione a quantidade de " \
        f"{'dias' if intervalo == 'Diário' else 'semanas'} para segurar a carteira",
        min_value=0,
        value=200,
        max_value=200,
        step=1
        )
    st.sidebar.divider()

    # ---------------------------------------------------

    # ---------------------------------------------------
    # cria um widget 'slider' para selecionar a quantidade de maiores médias de variações
    qtd_maiores_medias = st.sidebar.slider(
        label=f"Selecione a quantidade de maiores médias de retornos ({intervalo})",
        min_value=0,
        value=5,
        max_value=50,
        step=1
        )
    st.sidebar.divider()
    # ---------------------------------------------------

    # ---------------------------------------------------
    # cria um widget 'slider' para selecionar a quantidade de bebados para o backteste
    qtd_bebados = st.sidebar.slider(label="Selecione a quantidade de bebados",
                                    min_value=0,
                                    value=5,
                                    max_value=100,
                                    step=1)

    # cria um botão para rodar os backtestes
    botao = st.sidebar.button(label="Rodar Backtestes")

    if botao == True:

        # resgata o código da ação do índice do país selecionado
        simbolo_index = simbolos[paises[pais]][0]

        # roda os backtestes
        resultados_backtestes = \
            rodar_backtestes(
                acoes_selecionadas=acoes_selecionadas,
                data_inicial_bt=data_inicial,
                data_final_bt=data_final,
                intervalo=intervalos[intervalo],
                cotacoes_anteriores=qtd_cotacoes_anteriores,
                cotacoes_segurar=qtd_cotacoes_segurar,
                maiores_medias=qtd_maiores_medias,
                qtd_bebados=qtd_bebados,
                simbolo_index=simbolo_index
            )
        
        # resgada os patrimônios acumulados do moneta, do índice e dos bebados
        patrimonio_acumulado_moneta = \
            resultados_backtestes["acumulados"]["moneta"]
        
        patrimonio_acumulado_index = \
            resultados_backtestes["acumulados"]["index"]
        
        patrimonio_acumulado_bebados = \
            resultados_backtestes["acumulados"]["bebados"]
        
        # cria um gráfico com os patrimônios acumulados
        fig = go.Figure()

        # adiciona as linhas dos patrimônios acumulados do moneta
        fig.add_trace(
            go.Scatter(x=patrimonio_acumulado_moneta.index,
                       y=patrimonio_acumulado_moneta,
                       name="Moneta", line=dict(color="blue"))
            )
        
        # adiciona a linha do patrimônio acumulado do índice
        fig.add_trace(
            go.Scatter(x=patrimonio_acumulado_index.index,
                       y=patrimonio_acumulado_index,
                       name=simbolo_index, line=dict(color="red"))
            )
        
        # percorrendo os patrimônios acumulados dos bebados e adicionando ao gráfico
        for indice_bebado in range(qtd_bebados):
            fig.add_trace(

                go.Scatter(x=patrimonio_acumulado_bebados[indice_bebado].index,
                           y=patrimonio_acumulado_bebados[indice_bebado],
                           name=f"Bebado {indice_bebado}", line=dict(color="green")
                           )
                        )
        
        # exibe o gráfico com todos os patrimônios acumulados
        st.plotly_chart(fig)

        # resgata as variações periódicas do moneta
        variacoes_periodicas_moneta = \
            resultados_backtestes["variacoes"]["moneta"]
        
        # resgata as variações periódicas do índice
        variacoes_periodicas_index = \
            resultados_backtestes["variacoes"]["index"]

        # cria um objeto PerformanceTracker para calcular as métricas de performance
        tracker = PerformanceTracker(
            data_returns=variacoes_periodicas_moneta,
            market_returns=variacoes_periodicas_index,
            annual_risk_free=0.10,
            period=intervalos[intervalo]
        )
        
        # calcula as métricas de performance (sharpe, beta, retorno anual e max drawdown)
        sharpe = tracker.sharpe_ratio()
        beta = tracker.portfolio_beta()
        retorno_anual = tracker.annualized_return()
        max_drawdown = tracker.max_drawdown()
        
        # calcula o resultado do patrimonio final do moneta 
        resultado_final_moneta = \
            patrimonio_acumulado_moneta.iloc[-1] - 1

        # calcula o resultado do patrimonio final do índice
        resultado_final_index = \
            patrimonio_acumulado_index.iloc[-1] - 1
        
        # calcula o delta entre o resultado final do moneta e do índice
        delta = resultado_final_moneta - resultado_final_index

        # cria 3 colunas para exibir as métricas de performance
        col1, col2, col3 = st.columns(3)

        # na primeira coluna, cria widgets 'metric' para exibir o retorno final do moneta no backteste
        col1.metric(label=f"Retorno Final Moneta",
                    value=f"{resultado_final_moneta:.2%}",
                    delta=f"{delta:.2%}")
        
        # na segunda coluna, cria widgets 'metric' para exibir o retorno final do índice no backteste
        col2.metric(label=f"Retorno Final {simbolo_index}",
                    value=f"{resultado_final_index:.2%}")
        
        # na terceira coluna, cria widgets 'metric' para exibir o beta
        col3.metric(label=f"Beta",
                    value=f"{beta:.2f}")
        
        # cria 3 colunas para exibir as métricas de performance
        col4, col5, col6 = st.columns(3)

        # na primeira coluna, cria widgets 'metric' para exibir o sharpe
        col4.metric(label=f"Sharpe",
                    value=f"{sharpe:.2f}")
        
        # na segunda coluna, cria widgets 'metric' para exibir o max drawdown
        col5.metric(label=f"Max Drawdown",
                    value=f"{max_drawdown:.2f}%")
        
        # na terceira coluna, cria widgets 'metric' para exibir o retorno anual do moneta
        col6.metric(label=f"Retorno Anual",
                    value=f"{retorno_anual:.0f}%")
        
        # cria um subtitulo para exibir cada carteira gerada no backteste
        st.subheader("Carteiras geradas no Backteste")
        st.divider()

        # resgata os dados gerais do backteste (moneta, index) para cada carteira
        dados_gerais = resultados_backtestes["dados"]
        qtd_carteiras = len(dados_gerais)

        qtd_venceu_index = 0
        qtd_positivo = 0
        for i in range(qtd_carteiras):

            # dados do jogo do moneta
            dados_moneta = dados_gerais[i]["moneta"]

            # resgata os dados do moneta
            data_inicial = dados_moneta["data_inicio"]
            data_final = dados_moneta["data_fim"]
            carteira_otima = dados_moneta["carteira"]
            retornos_moneta: pd.Series = dados_moneta["retornos"]
            retorno_esperado = dados_moneta["retorno_esperado"]

            # resgata os dados do índice
            dados_index = dados_gerais[i]["index"]
            retornos_index: pd.Series = dados_index["retornos"]

            # calcula os retornos acumulados do moneta e do índice
            ret_acum_moneta = (1 + retornos_moneta).cumprod()
            ret_acum_index = (1 + retornos_index).cumprod()

            # calcula os retornos obtidos do moneta e do índice
            # ao final do backteste
            retorno_obtido_moneta = ret_acum_moneta.iloc[-1] - 1
            retorno_obtido_index = ret_acum_index.iloc[-1] - 1

            # exibe os dados da carteira
            st.subheader(f"Carteira {i + 1}")
            st.write(f"{data_inicial:%d-%m-%Y} até {data_final:%d-%m-%Y}")

            carteira_otima.name = "Percs"
            carteira_otima = carteira_otima.round(2) * 100
            carteira_otima = carteira_otima[carteira_otima > 1]

            # cria 4 colunas para exibir os dados da carteira
            col1, col2, col3, col4 = st.columns(4)
            col1.dataframe(carteira_otima)
            col2.metric(label="Retorno Esperado",
                        value=f"{retorno_esperado:.2%}")
            col3.metric(label="Retorno Moneta",
                        value=f"{retorno_obtido_moneta:.2%}")
            col4.metric(label=f"Retorno {simbolo_index}",
                        value=f"{retorno_obtido_index:.2%}")

            # cria um gráfico com os retornos acumulados do moneta e do índice
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=ret_acum_moneta.index,
                           y=ret_acum_moneta,
                           name="Moneta", line=dict(color="blue")))
            
            fig.add_trace(
                go.Scatter(x=ret_acum_index.index,
                           y=ret_acum_index,
                           name=f"{simbolo_index}",
                           line=dict(color="red")))
            
            # exibe o gráfico com os retornos acumulados
            st.plotly_chart(fig)

            # verifica se o retorno obtido do moneta foi positivo
            if retorno_obtido_moneta > 0:
                qtd_positivo += 1
            
            # verifica se o retorno obtido do moneta foi maior que o do índice
            if retorno_obtido_moneta > retorno_obtido_index:
                qtd_venceu_index += 1
            
            st.divider()

        # ao final de mostrar os resultados de cada carteira, exibe os resultados gerais
        st.subheader("Resultados Gerais")

        # cria 2 colunas para exibir os resultados gerais
        col1, col2 = st.columns(2)

        # na primeira coluna, cria um widget 'metric' para exibir a quantidade de carteiras
        # com retorno positivo
        col1.metric(label=f"Quantidade de carteiras com retorno positivo",
                    value=f"{qtd_positivo}/{qtd_carteiras}",
                    delta=f"{qtd_positivo / qtd_carteiras:.2%}")
        
        # na segunda coluna, cria um widget 'metric' para exibir a quantidade de carteiras
        # que venceram o índice
        col2.metric(
                    label=f"Quantidade de carteiras que venceram o {simbolo_index}",
                    value=f"{qtd_venceu_index} / {qtd_carteiras}",
                    delta = f"{qtd_venceu_index /qtd_carteiras:.2%}")
        