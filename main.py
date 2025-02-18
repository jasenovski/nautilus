import streamlit as st
from paginas.moneta import pagina_moneta
from paginas.backtestes import pagina_backtestes
from simbolos import simbolos

# cria um dicionário com as páginas disponíveis com os nomes das páginas como chave 
# e as funções como valor
paginas = {
    "Moneta": pagina_moneta,
    "Backtestes": pagina_backtestes
}

paises = {"Brasil": "BR", "EUA": "US"}
intervalos = {"Diário": "d", "Semanal": "w"}

def main():

    # Título da página
    st.sidebar.title(body="Menu")

    # Cria um selectbox para selecionar a página desejada
    modelo_selecionado = st.sidebar.selectbox(label="Selecione a página desejada",
                                              options=paginas.keys())

    # Chama/invoca a página desejada
    paginas[modelo_selecionado](simbolos, paises, intervalos)
     

if __name__ == "__main__":
    main()




