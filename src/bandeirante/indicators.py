import pandas as pd
import numpy as np

from bandeirante.volatility_lib import parkinson

def relative_strength_index(dataset, periods_ret = 1, RS_period = 60, close="PREULT"):
    
    ret = dataset[close].pct_change(periods_ret)
    
    G =  ret > 0
    L =  ret <= 0

    gainsS = np.zeros(dataset.shape[0])
    lossesS = np.zeros(dataset.shape[0])

    gainsS[G] = ret.loc[G]
    lossesS[L] = ret.loc[L]

    relative_strength = pd.DataFrame(
        np.array([gainsS,np.abs(lossesS)]).T,
        columns=["G","L"],
        index=dataset.index
        )

    relative_strength = relative_strength.ewm(span=RS_period).mean()
    relative_strength = (relative_strength["G"]/relative_strength["L"])

    return relative_strength/(1+relative_strength)

def average_true_range(
        dataset,
        alphaATR=0.1,
        open="PREABE",
        high="PREMAX",
        low="PREMIN",
        close="PREULT"
        ):
    true_range = np.max(
    [(dataset[high]-dataset[low]).abs().to_numpy(),
    (dataset[high]-dataset[close]).abs().to_numpy(),
    (dataset[low]-dataset[close]).abs().to_numpy()],axis=0
    )

    return pd.Series(true_range,index=dataset.index).ewm(alpha=alphaATR).mean()
     
def log_ret(data : pd.Series):
    return data.pct_change().apply(np.log1p)

def signal_to_noise_ratio(data,w,close="close",high="high",low="low"):
    log_ret_result = log_ret(data[close]).rolling(w).mean()
    volatility = parkinson(data,w,maxStr=high,minStr=low)

    return log_ret_result/volatility
    

def relative_strength_levy(
        data : pd.Series,
        w
        ):
    return (data/data.ewm(w).mean()).sub(1)


def rolling_z_score(dataset, field = "Close", w_short = 11,w_long = 252):
    field_data = dataset[field]

    short_mean = field_data.ewm(span=w_short).mean()
    long_mean = field_data.ewm(span=w_long).mean()
    long_std = field_data.ewm(span=w_long).mean()

    return (short_mean-long_mean)/long_std
    
#estimador beta    
def slope(log_ret,n):

    V = np.arange(n) 
    
    k_mu = V.mean()
    k_var = V.var()

    weights = ((V - k_mu)/k_var)

    return np.convolve(log_ret, weights, mode="valid")


def hurst(log_ret,w):

    roll = log_ret.rolling(w)

    R = roll.max()-roll.min()

    S = roll.std()

    return (R/S).apply(lambda x: np.log(x)/np.log(w))