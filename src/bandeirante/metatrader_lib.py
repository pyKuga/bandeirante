
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from .online import *

from functools import wraps

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
       
    
    def _check_lib(self,frame):
        if self.lib is None:
             raise RuntimeError("MT5 is not avaliable!")
        
        if frame is None:
            frame = self.lib.TIMEFRAME_H1

        return frame
    
    def _process_rates(self,rates,raw=False):
        if rates is None:
            raw_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
            return raw_data
        else:
            if raw:
                return rates
            else:           
                direct = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']).from_records(rates)
                direct['time'] = pd.to_datetime(direct['time'], unit='s')
                direct.set_index('time', inplace=True)
                return direct
            
    def getting_data(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            frame = kwargs.get('frame')
            kwargs['frame'] = self._check_lib(frame)

            # 2. Executa a função principal (que agora só chama o MT5)
            rates = func(self, *args, **kwargs)

            # 3. Pós-processamento: Formata os dados
            raw = kwargs.get('raw', False)
            return self._process_rates(rates, raw=raw)
        
        return wrapper

    @getting_data
    def get_possible(self,ticker,frame=None,raw=False):
        return self.lib.copy_rates_from_pos(ticker,frame,0,1_000_000)

    @getting_data
    def get_data(self,ticker,start,end,frame=None,raw=False):
        return self.lib.copy_rates_range(ticker,frame, start, end)
            
    

    def send_order(self, symbol, order_type, volume=1.0,deviation_pp = 20):
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
            "deviation": deviation_pp,
            "type_time": self.lib.ORDER_TIME_GTC,
            "type_filling": self.lib.ORDER_FILLING_IOC,
        }

        result = self.lib.order_send(request)

        if result.retcode != self.lib.TRADE_RETCODE_DONE:
            print("Erro:", result.retcode)
            return False

        return True
    
    def open_position(self,symbol, pos_type, volume=1.0,deviation_pp = 20):
        success = self.send_order(symbol, "BUY",volume,deviation_pp)

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



