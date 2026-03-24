import numpy as np

def online_ewm(x,x_prev,alpha):
    if x_prev == None:
        return x
    else:
        return alpha*x+(1-alpha)*x_prev

def log_close_to_close(P,previous_close_price):
    if previous_close_price == None:
        return np.nan
    else:
        return np.log(P/previous_close_price)

def online_std(x,std,mu,alpha):
    mu = online_ewm(x,mu,alpha)

    var = (1-alpha)*(
        std**2 + alpha*(x-mu)**2
    )
    
    return np.sqrt(var)