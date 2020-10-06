import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from scipy.stats import norm
class MC_GBM_European(object):
    
    def __init__(self):
        # Basic data
        self._ticker = None
        self._end_date = date.today()
        self._start_date = self._end_date.replace(self._end_date.year-1)
        
        # returns 
        self._return = None
        self._adj_close = None
        
        # for GBM and black scholes
        self._So = None
        self._n_sims = None
        self._mu = None
        self._sigma = None
        self._T = None # time to expiration
        self._N = None
        self._gbmpath = None
        self._K = None
        self._r = None
        self._BS_prem = None
        self._MC_BS_prem = None
        
    def set_ticker(self,TICKER):
        self._ticker = TICKER
        
    def set_return(self):
        df = yf.download(self._ticker, start=self._start_date,end=self._end_date, adjusted=True)
        self._adj_close = df['Adj Close']
        self._return = self._adj_close.pct_change().dropna() # log return

    def get_return(self):
        return self._return
    
    def plot_return(self):
        print(f'Average return: {100 * self._return.mean():.2f}%')
        self._return.plot(title=f'{self._ticker} Price until expiration: {self._start_date} - {self._end_date}')
    
    def set_gbm_n_expiration(self,T): # T = time until expiration
        self._So = self._adj_close[-1]
        self._n_sims = 100
        self._mu = self._return.mean()
        self._sigma = self._return.std()
        self._T = T
        self._N = T
        
        
        dt = self._T /self._N
        dW = np.random.normal(scale = np.sqrt(dt),size=(self._n_sims, self._N))
        W = np.cumsum(dW, axis=1)
        time_step = np.linspace(dt, self._T, self._N)
        time_steps = np.broadcast_to(time_step, (self._n_sims, self._N))
        self._gbmpath = self._So * np.exp((self._mu - 0.5 *self._sigma ** 2) * time_steps+ self._sigma * W)
        self._gbmpath  = np.insert(self._gbmpath, 0, self._So, axis=1)
    
    def get_gbm(self):
        return self._gbmpath
    
    def plot_gbm(self):
        PLOT_TITLE = (f'{self._ticker} Simulation 'f'({self._start_date}:{self._end_date})')
        index  = [self._end_date.replace(self._end_date.day+i) for i in range(self._T+1)]
        gbm_simulations_df = pd.DataFrame(np.transpose(self._gbmpath),index=index)
        # plotting
        ax = gbm_simulations_df.plot(alpha=0.2, legend=False)
        line_1, = ax.plot(index, gbm_simulations_df.mean(axis=1),color='red')
        ax.set_title(PLOT_TITLE, fontsize=16)
        
    def set_strike_n_interest(self,strike,interest_rate):
        self._K = strike
        self._r = interest_rate
    # we have S0,K,r,sigma,T, we can compute bs
    def get_BS( self, type='call'):
        d1 = (np.log(self._So/self._K) + (self._r + 0.5 * self._sigma**2) * self._T)/(self._sigma * np.sqrt(self._T))
        d2 = (np.log(self._So/self._K) + (self._r - 0.5 * self._sigma**2) * self._T)/(self._sigma * np.sqrt(self._T))
        if type == 'call':
            val = (self._So*norm.cdf(d1, 0, 1)-self._K*np.exp(-self._r*self._T)*norm.cdf(d2, 0, 1))
        elif type == 'put':
            val = (self._K*np.exp(-self._r*self._T)*norm.cdf(-d2, 0, 1)-self._So *norm.cdf(-d1, 0, 1))
        self._BS_prem = val
        return self._BS_prem
    
    def get_MC_BS(self):
        discount_factor = np.exp(-1 * self._r * self._T)
        self._MC_BS_prem = discount_factor * np.average(np.maximum(0,self._gbmpath[:, -1] - self._K))
        return self._MC_BS_prem
 