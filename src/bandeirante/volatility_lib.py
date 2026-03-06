import numpy as np
import pandas as pd

def parkinson(data,N,max_str="PREMAX",min_str="PREMIN"):
    C = 1/(4*np.log(2))
    delta=data[max_str].apply(np.log)-data[min_str].apply(np.log)
    mean_C = C*delta.pow(2).rolling(N).sum()/N
    
    return (mean_C).pow(1/2)

def garman_klass(
        data,
        N,
        max_str="PREMAX",
        min_str="PREMIN",
        open_str="PREABE",
        close_str="PREULT"
        ):
    C1 = 1/2
    C2 = 2*np.log(2)-1
    

    high_low = np.log(data[max_str] / data[min_str])
    close_open = np.log(data[close_str] / data[open_str])

    values_to_roll = pd.Series(C1*high_low**2-C2*close_open**2)

    return values_to_roll.rolling(N).mean().pow(1/2)
    

def rogers_satchell(
        data,
        N,
        max_str="PREMAX",
        min_str="PREMIN",
        open_str="PREABE",
        close_str="PREULT"):
    
    first_log = np.log(data[max_str] / data[close_str])
    second_log = np.log(data[max_str] / data[open_str])

    third_log = np.log(data[min_str] / data[close_str])
    fourth_log = np.log(data[min_str] / data[open_str])

    values_to_roll = pd.Series(first_log*second_log+third_log*fourth_log)    

    return values_to_roll.rolling(N).mean().pow(1/2)

def yang_zhang(
        data,
        N,
        max_str="PREMAX",
        min_str="PREMIN",
        open_str="PREABE",
        close_str="PREULT"):
    
    o = np.log(data[open_str])
    h = np.log(data[max_str])
    l = np.log(data[min_str])
    c = np.log(data[close_str])

    overnight = (o-c.shift(1)).pow(2).rolling(N).mean()
    open_to_close = (c-o).pow(2).rolling(N).mean()

    roger_satchell_var = rogers_satchell(
        data,
        N,
        max_str=max_str,
        min_str=min_str,
        open_str=open_str,
        close_str=close_str
        ).pow(2) 
    
    k = (0.34)/(1.34+(N+1)/(N-1))

    yang_zhang_var = overnight+k*open_to_close+(1-k)*roger_satchell_var

    return yang_zhang_var.pow(1/2)




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

    