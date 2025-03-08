import yfinance as yf
from tqdm import tqdm
from datetime import datetime, timedelta
import requests
import pandas as pd

def busca_cotacoes(simbolos: list, intervalo: str, 
                   **kwargs) -> pd.DataFrame:

    """
    Função que busca as variações periódicas das ações

    Args:
    simbolos (list): Lista com os símbolos (tickers) das ações
    cotacoes_anteriores (int): Quantidade de cotações anteriores a serem buscadas para as variações das ações
    kwargs (dict): dicionário com as chaves 'cotacoes_anteriores' e 'cotacoes_segurar' OU 'data_inicio' e 'data_fim'

    Returns:
    variacoes (pd.DataFrame): DataFrame com as variações periódicas das ações
    """

    # data de hoje (formato datetime)
    hoje_dtm: datetime = datetime.today()

    cotacoes_anteriores = kwargs.get('cotacoes_anteriores', None)
    cotacoes_segurar = kwargs.get('cotacoes_segurar', None)

    if cotacoes_anteriores is not None and cotacoes_segurar is not None:
        # data de início da busca (data de hoje menos a quantidade de cotações anteriores)
        if intervalo == "d":
            # se o intervalo for diário, subtrai a quantidade de dias
            data_inicio: datetime = hoje_dtm - timedelta(days=cotacoes_anteriores)
            data_fim: datetime = hoje_dtm + timedelta(days=cotacoes_segurar)
        elif intervalo == "w":
            # se o intervalo for semanal, subtrai a quantidade de semanas
            data_inicio: datetime = hoje_dtm - timedelta(weeks=cotacoes_anteriores)
            data_fim: datetime = hoje_dtm + timedelta(weeks=cotacoes_segurar)
        
        # converte a data de início para string (aaaa-mm-dd)
        data_inicio: str = data_inicio.strftime('%Y-%m-%d')
        data_fim: str = data_fim.strftime('%Y-%m-%d')
    else:
        data_inicio = kwargs.get('data_inicio', None)
        data_fim = kwargs.get('data_fim', None)

        if data_inicio is None or data_fim is None:
            raise ValueError("É necessário fornecer os parametros 'cotacoes_anteriores' e 'cotacoes_segurar'.")

    # busca as cotações das ações para o intervalo especificado
    df_cotacoes: pd.DataFrame = yf.download(simbolos, 
                                         start=data_inicio, 
                                         end=data_fim,
                                         group_by='ticker', 
                                         auto_adjust=True, 
                                         progress=False)# ['Adj Close']
    
    df_cotacoes = df_cotacoes.unstack().reset_index(name="Valores")
    df_cotacoes = df_cotacoes[df_cotacoes["Price"] == "Close"].reset_index(drop=True)
    df_cotacoes = \
        df_cotacoes.pivot_table(index=["Date"], 
                                columns="Ticker", 
                                values="Valores")

    return df_cotacoes

def formata_cotacoes(cotacoes: pd.DataFrame, maiores_medias: int) -> pd.DataFrame:

    """
    Função que formata as cotações das ações para variações periódicas e filtra as ações com maiores médias de retorno

    Args:
    cotacoes (pd.DataFrame): DataFrame com as cotações das ações
    intervalo (str): Intervalo de busca das variações periódicas das ações. 'd' para diário, 'w' para semanal
    maiores_medias (int): Quantidade de ações com maiores médias de retorno a serem filtradas

    Returns:
    variacoes_intervaladas_filtradas (pd.DataFrame): DataFrame com as variações periódicas das ações filtradas
    """

    series_percentuais_na = cotacoes.isna().sum() / cotacoes.shape[0]
    acoes_excluir = series_percentuais_na[series_percentuais_na > 0.1].index.to_list()

    # elimina as colunas (ações) que possuem mais de 10% de valores nulos
    cotacoes.drop(columns=acoes_excluir, inplace=True)

    # preenche os valores nulos com o último valor válido
    cotacoes.ffill(inplace=True)

    # elimina as colunas (axis = 1: nome das ações) que possuem valores nulos para datas específicas dentro do intervalo de busca    
    cotacoes.dropna(axis=0, inplace=True)

    # filtra as variações periódicas das ações (a cada 5 dias ou todos os dias)
    # cotacoes_intervaladas: pd.DataFrame = \
    #     cotacoes.iloc[::5] if intervalo == "w" else cotacoes.iloc[::1]

    # calcula as variações diárias das ações e elimina as linhas com valores nulos.
    # valores nulos podem ocorrer quando a ação não possui cotação em um determinado dia
    variacoes: pd.DataFrame = cotacoes.pct_change().dropna()    

    if maiores_medias > 0:
        # calcula as médias dos retornos das ações
        medias: pd.Series = variacoes.mean(axis=0)

        # o método 'nlargest' está presente em qualquer objeto do tipo 'Series'. 
        # Esse método retorna outro 'Series' com os 'n' maiores valores
        acoes_maiores_medias: pd.Series = medias.nlargest(maiores_medias)

        # pega as ações com as maiores médias de retorno
        variacoes_filtradas: pd.DataFrame = variacoes.loc[:, acoes_maiores_medias.index]

        return variacoes_filtradas, acoes_excluir
    
    return variacoes, acoes_excluir