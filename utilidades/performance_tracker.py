import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


class PerformanceTracker:
    """
    Esta classe foi desenvolvida por Vitor Hideki para 
    calcular métricas clássicas de mercado para análise de 
    desempenho de uma carteira de investimentos.
    """

    # construtor da classe
    def __init__(self, data_returns, market_returns=None, 
                 annual_risk_free=0, alpha_var=0.05, period="d"):

        """
        Construtor da classe PerformanceTracker

        Args:
        data_returns (pd.Series): retornos periódicos da carteira
        market_returns (pd.Series): retornos periódicos do mercado
        annual_risk_free (float): taxa livre de risco anual
        alpha_var (float): nível de significância da VaR
        period (str): período de análise (d = diário, w = semanal)
        """

        # atributos da classe
        self.data_returns = data_returns
        self.market_returns = market_returns
        self.annual_risk_free = annual_risk_free
        self.alpha_var = alpha_var
        self.period = period
        
        # validação do período de análise
        if period not in ["d", "w"]:
            raise Exception("Period must be either daily or weekly.")

    # método da classe
    def annualized_return(self, data_returns=False):
        """
        Método que calcula o retorno anualizado da carteira

        Args:
        data_returns (bool): retornos periódicos da carteira
        """

        if isinstance(data_returns, bool):
            data_returns = self.data_returns
        cumulative_return = (1 + data_returns).prod()
        if self.period == "d":
            years = len(data_returns) / 252
        elif self.period == "w":
            years = len(data_returns) / 52
        else:
            raise Exception("Period must be either daily or weekly.")
        annual_return = (cumulative_return ** (1 / years)) - 1
        return 100 * annual_return

    # método da classe
    def annualized_std_return(self, data_returns=False):

        """
        Método que calcula o desvio padrão anualizado da carteira

        Args:
        data_returns (bool): retornos periódicos da carteira
        """

        if isinstance(data_returns, bool):
            data_returns = self.data_returns
        std_dev = data_returns.std()
        if self.period == "d":
            annual_std_dev = std_dev * np.sqrt(252)
        elif self.period == "w":
            annual_std_dev = std_dev * np.sqrt(52)
        else:
            raise Exception("Period must be either daily or weekly.")
        return 100 * annual_std_dev

    # método da classe
    def sharpe_ratio(self):
        """
        Método que calcula o índice de Sharpe da carteira
        """
        sharpe_ratio = (self.annualized_return() - self.annual_risk_free) / self.annualized_std_return()
        return sharpe_ratio

    # método da classe
    def max_drawdown(self):
        """
        Método que calcula o máximo drawdown da carteira
        """
        cumulative_return = np.cumprod(1 + self.data_returns)
        rolling_max = np.maximum.accumulate(cumulative_return)
        drawdown = (cumulative_return - rolling_max) / rolling_max
        max_drawdown = np.min(drawdown)
        return max_drawdown * 100

    # método da classe
    def portfolio_beta(self):
        """
        Método que calcula o beta da carteira
        """
        if isinstance(self.market_returns, pd.Series):
            covariance = np.cov(self.data_returns, self.market_returns, ddof=0)[0, 1]
            market_variance = np.var(self.market_returns, ddof=0)
            beta = covariance / market_variance
            return beta
        else:
            return None


    # o método mágico __call__ permite que a classe seja chamada como uma função
    # este método é invicado assim que a classe instancia um objeto.
    def __call__(self):
        result = {
            "sharpe": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "beta": self.portfolio_beta(),
            "annual_return": self.annualized_return(),
        }
        return result