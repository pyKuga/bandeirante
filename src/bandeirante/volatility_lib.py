import numpy as np
import pandas as pd


def get_engine_names(engine):

    names_dict = {

        "yfinance":{
            "close":"Close",
            "open":"Open",
            "low":"Low",
            "high":"High"
            },

        "mt5":{
            "close":"close",
            "open":"open",
            "low":"low",
            "high":"high"
        },

        "B3":{
            "close":"PREULT",
            "open":"PREABE",
            "low":"PREMIN",
            "high":"PREMAX"
        }
    }
    
    col_names =  names_dict[engine]

    C = col_names["close"]
    O = col_names["open"]
    L = col_names["low"]
    H = col_names["high"]

    return O,H,L,C
    


def parkinson(data,N,engine="yfinance"):


    O,H,L,C = get_engine_names(engine)

    CONSTANT = 1/(4*np.log(2))
    delta=data[H].apply(np.log)-data[L].apply(np.log)
    mean_C = CONSTANT*delta.pow(2).rolling(N).sum()/N
    
    return (mean_C).pow(1/2)

def garman_klass(
        data,
        N,
        engine="yfinance"
        ):
    
    O,H,L,C = get_engine_names(engine)

    C1 = 1/2
    C2 = 2*np.log(2)-1
    

    high_low = np.log(data[H] / data[L])
    close_open = np.log(data[C] / data[O])

    values_to_roll = pd.Series(C1*high_low**2-C2*close_open**2)

    return values_to_roll.rolling(N).mean().pow(1/2)
    

def rogers_satchell(
        data,
        N,
        engine="yfinance"):
    
    O,H,L,C = get_engine_names(engine)
    
    first_log = np.log(data[H] / data[C])
    second_log = np.log(data[H] / data[O])

    third_log = np.log(data[L] / data[C])
    fourth_log = np.log(data[L] / data[O])

    values_to_roll = pd.Series(first_log*second_log+third_log*fourth_log)    

    return values_to_roll.rolling(N).mean().pow(1/2)


def yang_zhang(
        data,
        N,
        engine="yfinance"
        ):
    
    O,H,L,C = get_engine_names(engine)
    
    o = np.log(data[O])
    h = np.log(data[H])
    l = np.log(data[L])
    c = np.log(data[C])

    overnight = (o-c.shift(1)).pow(2).rolling(N).mean()
    open_to_close = (c-o).pow(2).rolling(N).mean()

    roger_satchell_var = rogers_satchell(
        data,
        N,engine=engine
        ).pow(2) 
    
    k = (0.34)/(1.34+(N+1)/(N-1))

    yang_zhang_var = overnight+k*open_to_close+(1-k)*roger_satchell_var

    return yang_zhang_var.pow(1/2)




def pct_change_day(data,n,engine="yfinance"):

    O,H,L,C = get_engine_names(engine)

    openData = data[O].ewm(span=n).mean()
    closeData = data[C].ewm(span=n).mean()

    return closeData/openData-1


def hankel_matrix(data, p):
    data = np.asarray(data).flatten()
    n = len(data)
    return np.column_stack(
        [data[p-i-1:n-i-1] for i in range(p)]
    )

    