import pandas as pd

def GetSeries(caminho,freq="D"):
    BCSeries = pd.read_json(path_or_buf=caminho)
    BCSeries["data"] = pd.to_datetime(BCSeries["data"],format="%d/%m/%Y")
    BCSeries.set_index(BCSeries["data"],inplace=True,drop=True)
    BCSeries.drop(columns="data",inplace=True)
    
    return BCSeries

