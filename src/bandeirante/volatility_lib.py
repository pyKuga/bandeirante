import numpy as np
import pandas as pd

def parkinson(data,N,maxStr="PREMAX",minStr="PREMIN"):
    C = 1/(4*np.log(2))
    delta=data[maxStr].apply(np.log)-data[minStr].apply(np.log)
    mean_C = C*delta.pow(2).rolling(N).sum()/N
    
    return (mean_C).pow(1/2)

def garman_klass(data,N,maxStr="PREMAX",minStr="PREMIN",openStr="PREABE",closeStr="PREULT"):
    C = 2*np.log(2)-1
    
    H=data[maxStr].apply(np.log).rolling(N).sum()
    L=data[minStr].apply(np.log).rolling(N).sum()
    C=data[closeStr].apply(np.log).rolling(N).sum()
    O=data[openStr].apply(np.log).rolling(N).sum()

    return (((H-L).pow(2)-C*(C-O).pow(2))/N).pow(1/2)

def RogersSatchell():
    pass

def pct_change_day(data,n):
    openData = data["open"].ewm(span=n).mean()
    closeData = data["close"].ewm(span=n).mean()

    return closeData/openData-1


def hankel_matrix(data, p):
    data = np.asarray(data).flatten()
    n = len(data)
    return np.column_stack(
        [data[p-i-1:n-i-1] for i in range(p)]
    )

    