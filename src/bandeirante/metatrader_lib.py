
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta



try: 
    import MetaTrader5 as mt5   
    mt5.initialize()


    def MQLDataGet(ticker,start,end,frame=mt5.TIMEFRAME_H1):
            
        
            rates = mt5.copy_rates_range(ticker,frame, start, end)
            raw_data = pd.DataFrame(rates)
            raw_data['time'] = pd.to_datetime(raw_data['time'], unit='s')
            raw_data.set_index('time', inplace=True)
            #raw_data.to_csv(ticker+".csv")
            return raw_data
except:
    print("Esse PC não tem MT5")

