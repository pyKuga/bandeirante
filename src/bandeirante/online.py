import numpy as np
import pandas as pd



class DataStructureForThisBot():  

    def __init__(self,ordinal_classfier,event_detector,w=12):
        
        self.ORIGINAL_COLUMNS =  [
            'open', 
            'high', 
            'low', 
            'close', 
            'tick_volume', 
            'spread', 
            'real_volume',
            ]
        
        self.AUXILIARY_COLUMNS = [
            "log_ret",
            "events"
        ]

        self.AUTOREG_FEATURES = [
            "z_noise",
            "patterns",
            "loc_vol"
        ]


        #DESENVOLVER A PARTE DE JANELAS AUTOREGRESSIVAS
        self.auto_reg = np.zeros((len(self.AUTOREG_FEATURES),self.ordinal_classfier.pattern_length))

        temp_autoreg = []
        for el in self.AUTOREG_FEATURES:
            for i in range(0,self.ordinal_classfier.pattern_length):
                temp_autoreg.append(el+f"_{i}")

        self.AUTOREG_FEATURES = temp_autoreg


        self.SINGLE_FEATURES = [
            "simple_vol",
            "relative_return",
            "S_pos",
            "S_neg"
        ]

       
        self.FEATURES_COLUMNS = self.AUTOREG_FEATURES + self.SINGLE_FEATURES
            

        
        self.META_FEATURES = [
            'day',
            'hour',
            'weekday',
            'seconds_to_close',
            'labels',
            'candle_size',
            'candle_proportion',
            'prediction_entropy'
            
        ]
        
        self.COLUMNS_ON_DATAFRAME = self.ORIGINAL_COLUMNS + self.FEATURES_COLUMNS + self.META_FEATURES + self.AUXILIARY_COLUMNS
        
        self.ticker_data = pd.DataFrame(columns=self.COLUMNS_ON_DATAFRAME)


        self.ordinal_classfier = ordinal_classfier
        self.event_detector = event_detector

        self.window = np.ones(self.ordinal_classfier.pattern_length)*np.nan
        
        self.alpha = 2/(1+w)
        

        self.H = 1

        self.mu = None
        
        self.std = 0
        self.previous_close_price = None

        self.first_time = None
        self.last_time = None

        
   

    def reset_values(self):
        self.mu = None
        
        self.std = 0
        self.previous_close_price = None

        self.first_time = None
        self.last_time = None


    def candle_processing(self,candle):

        info_as_dict = dict(zip(self.ORIGINAL_COLUMNS,candle[1:]))

        instant_time = pd.Timestamp(candle[0],unit="s")

        if self.first_time == None:
            self.first_time = instant_time

        info_as_dict["day"] = instant_time.day
        info_as_dict["hour"] = instant_time.hour
        info_as_dict["weekday"] = instant_time.weekday()

        info_as_dict["seconds_to_close"] = (instant_time-self.first_time).seconds


        if self.previous_close_price == None:
            info_as_dict["log_ret"] = 0
        else:
            info_as_dict["log_ret"] = np.log(info_as_dict["close"]/self.previous_close_price)

        self.previous_close_price = info_as_dict["close"]       

        info_as_dict["loc_vol"] = self.online_std(info_as_dict["log_ret"])


        info_as_dict["z_noise"] = info_as_dict["log_ret"]/(info_as_dict["loc_vol"]+1e-10)

        info_as_dict["candle_size"]= np.abs((info_as_dict["close"]-info_as_dict["open"]))
        info_as_dict["candle_proportion"]= np.abs(info_as_dict["close"]-info_as_dict["open"])/np.abs(info_as_dict["high"]-info_as_dict["low"]+1e-10)

        info_as_dict["simple_vol"] = np.log(info_as_dict["high"]/info_as_dict["low"])
        info_as_dict["relative_return"] = np.log(info_as_dict["close"]/info_as_dict["open"])

        info_as_dict["labels"] = 2
            
        

        self.window[:-1] = self.window[1:]
        self.window[-1] = info_as_dict["log_ret"]


        if np.isnan(self.window).sum() == 0:
            info_as_dict["patterns"] = self.ordinal_classfier.check_pattern(self.window)
            #info_as_dict["entropy"] = ordinal_classfier.entropy(info_as_dict["patterns"])
        else:
            info_as_dict["patterns"] = 0

        info_as_dict["time"] = instant_time

        info_as_dict["prediction_entropy"] = 0.0

        return info_as_dict


    def online_std(self,x):

        if self.mu == None:
            self.mu = x
        else:
            self.mu = self.alpha*x+(1-self.alpha)*self.mu

        x_2 = (x-self.mu)**2

        var = (1-self.alpha)*(self.std**2 + self.alpha*x_2)
        
        self.std =  np.sqrt(var)


        return self.std
    
    def decision_to_trade(self,BSN_model,meta_model):
        
        X = self.ticker_data.loc[[self.last_time], self.FEATURES_COLUMNS]

        self.probs_BSN = BSN_model.predict_proba(X)

        self.ticker_data.loc[self.last_time,"labels"] = BSN_model.predict(X)
        self.ticker_data.loc[self.last_time,"prediction_entropy"] = -(np.log(self.probs_BSN)*self.probs_BSN).sum(axis=1)/np.log(2)

        meta_info = self.ticker_data.loc[[self.last_time],self.META_FEATURES]

        self.probs_meta = meta_model.predict_proba(meta_info)

        self.ticker_data.loc[self.last_time,"meta"] = meta_model.predict(meta_info)


        self.H = -(np.log(self.probs_meta)*self.probs_meta).sum(axis=1)/np.log(2)

    def update_ticker_dataset(self,info_as_dict):
        self.ticker_data.loc[info_as_dict["time"],self.COLUMNS_ON_DATAFRAME] = info_as_dict

