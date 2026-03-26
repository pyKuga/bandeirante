import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from bandeirante.indicators import signal_to_noise_ratio, log_ret
from bandeirante.volatility_lib import get_engine_names

class CUSUM:
    def __init__(self, k_up=1, k_down=1, h_min = 1e-3):
        self.k_up = k_up
        self.k_down = k_down
        self.h_min = h_min

        self.Sp_k = 0 #S^{+}_{k}
        self.Sp_k_1 = 0 #S^{+}_{k-1}

        self.Sn_k = 0 #S^{-}_{k}
        self.Sn_k_1 = 0 #S^{-}_{k-1}

        self.S_pos = 0
        self.S_neg = 0

    def reset(self):
        self.Sp_k = 0 #S^{+}_{k}
        self.Sp_k_1 = 0 #S^{+}_{k-1}

        self.Sn_k = 0 #S^{-}_{k}
        self.Sn_k_1 = 0 #S^{-}_{k-1}

        self.S_pos = 0
        self.S_neg = 0

    def event_detection(self,x,sigma):

        event = None 

        self.Sp_k = np.maximum(x+self.Sp_k_1,0)
        self.Sn_k = np.minimum(x+self.Sn_k_1,0)

        self.S_pos = self.Sp_k
        self.S_neg = self.Sn_k

        if self.Sp_k > np.maximum(self.h_min,self.k_up*sigma):
            self.Sp_k = 0
            event = True
        
        elif self.Sn_k < -np.maximum(self.h_min,self.k_down*sigma):
            self.Sn_k = 0
            event = True
        else:            
            event = False

        self.Sp_k_1 = self.Sp_k
        self.Sn_k_1 = self.Sn_k

        return event           
       

    def detect_on_series(self,data,vol_hist):

        events = np.zeros(data.shape,dtype=bool)

        if np.isscalar(vol_hist):
            loc_vol = np.full_like(data, vol_hist, dtype=float)
        else:
            loc_vol = pd.Series(vol_hist).to_numpy()

        for i,el in enumerate(data.to_numpy()):
            sigma = loc_vol[i]
            events[i] = self.event_detection(el,sigma)
        
        return events


def time_windows(vol_hist):
    if np.isscalar(vol_hist):
        return 1/vol_hist
    else:
        return (1/vol_hist).apply(np.ceil).astype(int)
    
    

def triple_barrier(
        dataset,
        vol_hist,
        events,
        buy_label = 1,
        neutral_label = 2,
        sell_label = 0,
        engine="yfinance",
        **timedelta_kwargs
        ):
    
    O,H,L,C = get_engine_names(engine)

    labels = pd.Series(
        np.ones(dataset.shape[0])*neutral_label,
        index=dataset.index)

    events_indexes = dataset.index[events]

    

    if np.isscalar(vol_hist):
        loc_vol = pd.Series(vol_hist,index=dataset.index)
    else:
        loc_vol = pd.Series(vol_hist)

    #windows_T = time_windows(loc_vol)

    for event in events_indexes:
        #window_barrier = np.minimum(maximum_window,windows_T[event])

        sigma_event = loc_vol[event]

        price_event = dataset.loc[event,C]

        upper_barrier = price_event*(1+sigma_event)
        inferior_barrier = price_event*(1-sigma_event)
        

        time_window = dataset.loc[event:event+pd.Timedelta(**timedelta_kwargs),[H,L]]

        upper_trepassed = (time_window[H] > upper_barrier)
        lower_trepassed = time_window[L] < inferior_barrier

        upper_index = upper_trepassed.idxmax() if upper_trepassed.any() else None
        lower_index = lower_trepassed.idxmax() if lower_trepassed.any() else None


        if upper_index is None and lower_index is None:
            continue
        elif (
            (lower_index is None) or (upper_index is not None and upper_index > lower_index)
            ):
            labels.loc[event] = buy_label 
        else:
            labels.loc[event] = sell_label

    return labels

def vol_to_time(vol_hist):
    return (1/vol_hist).apply(np.ceil).astype(int)

def return_on_prediction(log_ret, X,vol_hist,y_pred):
    local_index = X.index
    
    T = vol_to_time(vol_hist)

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
        
