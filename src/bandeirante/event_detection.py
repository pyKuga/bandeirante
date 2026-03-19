import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from bandeirante.indicators import signal_to_noise_ratio, log_ret
from bandeirante.volatility_lib import get_engine_names

def CUSUM_event_detection(
        log_ret : pd.Series | np.ndarray,
        vol_hist,
        k_up=1, 
        k_down=1,  
        h_min = 1e-3
        ):

    log_returns = log_ret.fillna(0).to_numpy()
    
    if np.isscalar(vol_hist):
        loc_vol = np.full_like(log_returns, vol_hist, dtype=float)
    else:
        loc_vol = pd.Series(vol_hist).to_numpy()

    S_pos = np.zeros(log_returns.shape)
    S_neg = np.zeros(log_returns.shape)

    cusum_events = np.zeros(log_returns.shape).astype(bool)

    for i in range(1,log_returns.shape[0]):
        S_pos[i] = np.maximum(log_returns[i]+S_pos[i-1],0)
        S_neg[i] = np.minimum(log_returns[i]+S_neg[i-1],0)

        if S_pos[i] > np.maximum(h_min,k_up*loc_vol[i]):
            cusum_events[i] = True
            S_pos[i] = 0
        
        elif S_neg[i] < -np.maximum(h_min,k_down*loc_vol[i]):
            cusum_events[i] = True
            S_neg[i] = 0

    return cusum_events


def time_windows(vol_hist):
    return (1/vol_hist).apply(np.ceil).astype(int)
    

def triple_barrier(
        dataset,
        vol_hist,
        events,
        maximum_window = 20,
        buy_label = 1,
        neutral_label = 2,
        sell_label = 0,
        engine="yfinance"
        ):
    
    O,H,L,C = get_engine_names(engine)

    

    dataset["label"] = neutral_label

    events_indexes = dataset.index[events]

    windows_T = time_windows(vol_hist)

    for event in events_indexes:
        window_barrier = np.minimum(maximum_window,windows_T[event])

        sigma_event = vol_hist[event]

        price_event = dataset.loc[event,C]

        upper_barrier = price_event*(1+sigma_event)
        

        inferior_barrier = price_event*(1-sigma_event)
        

        time_window = dataset.loc[event:event+pd.Timedelta(days=window_barrier),[H,L]]

        upper_trepassed = (time_window[H] > upper_barrier)
        lower_trepassed = time_window[L] < inferior_barrier

        upper_index = upper_trepassed.idxmax() if upper_trepassed.any() else None
        lower_index = lower_trepassed.idxmax() if lower_trepassed.any() else None


        if upper_index is None and lower_index is None:
            continue
        elif (
            (lower_index is None) or (upper_index is not None and upper_index > lower_index)
            ):
            dataset.loc[event,"label"] = buy_label 
        else:
            dataset.loc[event,"label"] = sell_label

    return dataset

def vol_to_days(vol_hist):
    return (1/vol_hist).apply(np.ceil).astype(int)

def return_on_prediction(log_ret, X,vol_hist,y_pred):
    local_index = X.index
    
    T = vol_to_days(vol_hist)

    deltaT = T.loc[local_index].apply(lambda x: pd.Timedelta(days=x))

    final_test = local_index + deltaT

    index_list = list(zip(local_index,final_test))

    returns_windows = np.zeros(y_pred.shape)

    for i ,tupla in enumerate(index_list):
        ret = log_ret.loc[tupla[0]:tupla[1]].sum()
        if y_pred[i] == 0:
            returns_windows[i] = -1*ret
        else:
            returns_windows[i] = ret   

    return returns_windows

def HMM_state_detection(
        dataset, 
        engine="yfinance",
        w = 52,
        p=2,
        train_percentual = 0.8, 
        **hmm_kwargs
        ):
    
    O,H,L,C = get_engine_names(engine)
    
    model_hmm  = GaussianHMM(n_components=p,**hmm_kwargs)

    snr_np = signal_to_noise_ratio(dataset,w,close=C,high=H,low=L).dropna().to_numpy().reshape(-1,1)
    
    train_entries = np.ceil(snr_np.shape[0]*train_percentual).astype(int)

    train_data = snr_np[:train_entries]

    model_hmm.fit(train_data)

    state_prediction = model_hmm.predict(snr_np)


    dataset["State"] = 0
    dataset.loc[dataset.index[w:],"State"] = state_prediction

    log_returns = log_ret(dataset["Close"]).fillna(0)

    if (log_returns[(dataset["State"] == 1)].sum() < 0):
        dataset["State"] = (~dataset["State"].astype(bool)).astype(int)
        
