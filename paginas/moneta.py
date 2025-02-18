import streamlit as st
from cotacoes.cotacoes import busca_cotacoes, formata_cotacoes
from modelo.moneta import moneta_ag
from utilidades.gerais import gera_df_carteira, obter_data_vender
import plotly.graph_objects as go
from datetime import datetime

def pagina_moneta(simbolos: dict, paises: dict, intervalos: dict) -> None:

    """
    Página que contém a aplicação do modelo Moneta.

    Args:
    simbolos (dict): Dicionário com os símbolos das ações de cada país.
    paises (dict): Dicionário com os países disponíveis.
    intervalos (dict): Dicionário com os intervalos disponíveis.

    Returns:
    None
    """

    # cria o título da página
    st.title(body="Modelo Moneta")
    st.write("O moneta é uma ferramenta quantitativa para diversificação de carteira.")
    st.divider()

    # ---------------------------------------------------
    # cria duas colunas no sidebar para inserir os widgets
    colunas = st.sidebar.columns(2)

    # na primeira coluna, cria um widget 'radio' para selecionar o país
    pais = colunas[0].radio(label="Selecione uma bolsa de ações", options=paises, index=0)
    moeda = f"{'R$' if pais == 'Brasil' else 'US$'}"
    
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
        # do país selecionado (sem o código do índice do país)
        bolsa_acoes = simbolos[paises[pais]][1:6]
    
    # cria um widget 'multiselect' para apresentar as ações possíveis de serem selecionadas
    # para rodar o modelo
    acoes_selecionadas = st.sidebar.multiselect(label="Selecione as ações para rodar o modelo",
                                                options=simbolos[paises[pais]][1:],
                                                default=bolsa_acoes)

    st.sidebar.divider()
    # ---------------------------------------------------

    # ---------------------------------------------------
    # cria um widget 'number_input' para inserir o valor do investimento
    valor_investimento = st.sidebar.number_input(label="Insira o valor do investimento:",
                                                 min_value=100,
                                                 value=1000,
                                                 step=50)
    st.sidebar.divider()
    # ---------------------------------------------------

    # ---------------------------------------------------
    # cria um widget 'slider' para selecionar o percentual mínimo 
    # para uma ação aparecer na carteira final
    percentual_filtrar = st.sidebar.slider(
        label="Selecione o percentual mínimo para uma ação aparecer na carteira",
        min_value=0,
        value=1,
        max_value=5,
        step=1
    )

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

    # cria um widget 'slider' para selecionar a quantidade de maiores médias de retornos
    qtd_maiores_medias = st.sidebar.slider(
        label=f"Selecione a quantidade de maiores médias de retornos ({intervalo})",
        min_value=0,
        value=5,
        max_value=50,
        step=1
    )
    st.sidebar.divider()
    # ---------------------------------------------------

    # cria um botão para rodar o modelo
    botao = st.sidebar.button(label="Rodar Moneta")

    if botao == True:
        
        print("Baixando as cotações das ações selecionadas...")
        cotacoes = busca_cotacoes(simbolos=acoes_selecionadas,
                                  cotacoes_anteriores=qtd_cotacoes_anteriores,
                                  cotacoes_segurar=0,
                                  intervalo=intervalos[intervalo])
        
        if cotacoes.isna().sum().sum() > 0:
            acoes_com_na = cotacoes.columns[cotacoes.isna().sum() > 0]
            mensagem = f"Erro: as ações {list(acoes_com_na)} possuem valores nulos. " \
                        "Por favor, atualize/retire estas ações no arquivo dos símbolos."
            st.warning(mensagem)
            return
        
        # acima, o 'cotacoes_segurar' é 0, pois o modelo vai rodar com dados atuais
        # e não em um backtest
        
        if cotacoes.empty:
            st.warning(""":warning: Não foi possível baixar as cotações das ações selecionadas. 
                       Tente rodar o modelo novamente.""")
            return

        print("Formatando as cotações das ações selecionadas para variações...")
        variacoes, acoes_excluidas = formata_cotacoes(cotacoes=cotacoes,
                                                        maiores_medias=qtd_maiores_medias)
        
        if len(acoes_excluidas) > 0:
            st.warning(f":warning: As ações {acoes_excluidas} foram excluídas na preparação dos dados, pois possuem mais de 10% de valores nulos."
                       f"Por favor, atualize o ticker ou retire estas ações no arquivo dos símbolos.")
        
        print("Rodando o modelo Moneta com as variações formatadas...")
        carteira_otima = moneta_ag(variacoes=variacoes)

        print("Formatando a carteira ótima...")
        df_carteira = gera_df_carteira(carteira_final=carteira_otima,
                                       cotacoes=cotacoes,
                                       pais=paises[pais],
                                       percentual_filtrar=percentual_filtrar,
                                       valor_investir=valor_investimento)
        
        print("Mostrando a carteira final na tela...")
        
        if df_carteira is None:
            st.warning(""":warning: O filtro percentual retirou todas as ações da carteira final. 
                       Diminua o valor deste filtro.""")
            return

        # cria um subtitulo
        st.subheader("Carteira Final")
        # mostra o dataframe da carteira final na tela
        st.dataframe(df_carteira)


        valor_investir_final = df_carteira[f"Investido ({moeda})"].sum()
        colunas = st.columns(2)
        # cria um widget 'metric' para mostrar o valor total a investir
        # ao final do modelo
        colunas[0].metric(label="Valor total investir", 
                          value=f"{moeda} {valor_investir_final:.2f}")
        # cria um widget 'metric' para mostrar o percentual total investido do
        # valor de investimento. Por exemplo, se o valor de investimento for 1000
        # e o percentual total investido for 98%, então o valor total investido
        # será de 980
        colunas[1].metric(label="Perc total carteira", 
                          value=f"{df_carteira['Investido (%)'].sum():.1f}%")

        # resgata a data de hoje e calcula a data para vender a carteira
        hoje = datetime.today().strftime("%Y-%m-%d")
        data_vender = obter_data_vender(data_compra=hoje, 
                                        cotacoes_segurar=qtd_cotacoes_segurar, 
                                        intervalo=intervalos[intervalo])
        
        # cria um widget 'warning' para mostrar a data para vender a carteira
        st.warning(f":date: Vender a carteira aproximadamente em: **{data_vender}**")

        # cria um gráfico de pizza com a quantidade de ações na carteira final
        fig = go.Figure(data=go.Pie(labels=df_carteira.index, 
                                    values=df_carteira["Qtd de Acoes"]))

        # formata o gráfico de pizza
        fig.update_traces(hoverinfo="label+percent+value", # mostra label, percentual e valor ao passar o mouse
                          textinfo="label+percent",
                          textfont_size=12, 
                          textfont_family="Ubuntu",
                          marker={"line": {"color": "white", "width": 2}})

        # formata o layout do gráfico de pizza
        fig.update_layout(title="Gráfico de Pizza", 
                          title_font_size=30, 
                          title_font_family="Ubuntu", 
                          title_font_color="black")
        
        # mostra o gráfico de pizza na tela
        st.plotly_chart(fig)

        # cria um widget 'success' para mostrar que o modelo rodou com sucesso
        st.success(":tada: Modelo Moneta rodado com sucesso!")
