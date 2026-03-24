
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta

class MQLOperator:
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

    

    def send_order(self, symbol, order_type, volume=1.0):
        tick = self.lib.symbol_info_tick(symbol)

        if order_type == "BUY":
            price = tick.ask
            order_type_mt5 = self.lib.ORDER_TYPE_BUY
        else:
            price = tick.bid
            order_type_mt5 = self.lib.ORDER_TYPE_SELL

        request = {
            "action": self.lib.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price,
            "deviation": 20,
            "type_time": self.lib.ORDER_TIME_GTC,
            "type_filling": self.lib.ORDER_FILLING_IOC,
        }

        result = self.lib.order_send(request)

        if result.retcode != self.lib.TRADE_RETCODE_DONE:
            print("Erro:", result.retcode)
            return False

        return True
    
    def open_position(self,symbol, pos_type):
        success = self.send_order(symbol, "BUY")

        if success:
            return {"type": pos_type, "symbol": symbol}
        
        return None
    
    def close_position(self,pos):
        success = self.send_order(pos["symbol"], "SELL")
        return success

    def close_all_positions(self):
        positions = self.lib.positions_get()

        if positions is None:
            return

        for pos in positions:
            if pos.type == self.lib.POSITION_TYPE_BUY:
                self.send_order(pos.symbol, "SELL", pos.volume)
            else:
                self.send_order(pos.symbol, "BUY", pos.volume)
