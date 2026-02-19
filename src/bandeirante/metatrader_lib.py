
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class MQLData:
    def __init__(self):
        try: 
            import MetaTrader5 as mt5   
            if not mt5.initialize():
                raise RuntimeError("MT5 failed to initialize")
            self.lib = mt5
        except:
            self.lib = None
            print("This PC has no MT5 installed!")

    def get_data(self,ticker,start,end,frame=None):
        
        if self.lib is None:
             raise RuntimeError("MT5 is not avaliable!")
        
        if frame is None:
            frame = self.lib.TIMEFRAME_H1
            
        
        rates = self.lib.copy_rates_range(ticker,frame, start, end)
        raw_data = pd.DataFrame(rates)
        raw_data['time'] = pd.to_datetime(raw_data['time'], unit='s')
        raw_data.set_index('time', inplace=True)
        #raw_data.to_csv(ticker+".csv")
        return raw_data

    

