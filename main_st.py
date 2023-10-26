import streamlit as st
from paginas.pagina_nautilus import pag_naut
from paginas.pagina_backtestes import pag_bts


paginas = {"Nautilus": pag_naut,
           "Backtestes": pag_bts}

def main():
    modelo = st.sidebar.selectbox(label="Selecione a página desejada:", options=list(paginas.keys()), index=0)
    paginas[modelo]()


if __name__ == "__main__":
    main()