import plotly.graph_objects as go
import pandas as pd
import numpy as np

def parkinson_month(original_data,close="Close",open="Open",high="High",low="Low"):

    dataset = original_data.copy()

    dataset["pct"] = dataset[close].pct_change()
    dataset["dia"] = dataset.index.day
    dataset["mes"] = dataset.index.to_period("M")

    #parkinson vol

    out = pd.pivot_table(dataset,values=[high,low,close],index="mes",columns="dia")
    # vol_hist = out.std(axis=1)
    delta = (out[high].apply(np.log)-out[low].apply(np.log))

    C = 1/(4*np.log(2))

    vol_hist = (C*delta.pow(2).mean(axis=1)).apply(np.sqrt)

    #vol_hist = (vol_hist.pow(2).apply(np.exp)-1).apply(np.sqrt)

    vol_hist.index = vol_hist.index.to_timestamp()
    vol_hist = vol_hist.reindex(dataset.index, method="ffill")*np.sqrt(20)

    #vol_hist =dataset["PREULT"].pct_change().rolling(20).std().shift(1)

    ref_price = dataset[close].resample("ME").last().reindex(dataset.index, method="ffill")

    return vol_hist, ref_price

def create_observation(dataset,ticker=None,close="Close",open="Open",high="High",low="Low"):
    try:
        observation = dataset.xs(ticker,axis=1,level=0).copy()
    except:
        observation = dataset
    vol_hist, ref_price = parkinson_month(observation,close=close,open=open,high=high,low=low)
    observation["ref_price"] = ref_price#observation[close].ewm(span=61).mean()
    observation["minus_1sigma"] = observation["ref_price"]*(1-vol_hist)
    observation["minus_2sigma"] = observation["ref_price"]*(1-2*vol_hist)
    observation["plus_1sigma"] = observation["ref_price"]*(1+vol_hist)
    observation["plus_2sigma"] = observation["ref_price"]*(1+2*vol_hist)

    observation["percentual"] = ((observation[close]-observation["minus_2sigma"])/(observation["plus_2sigma"] - observation["minus_2sigma"]))

    return observation


def generate_plotly_graph(ticker,dataset):
    observation = create_observation(dataset,ticker)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=observation.index,y=observation["Close"],name="Close"))

    fig.add_trace(go.Scatter(
        x=observation.index,
        y=(observation["minus_1sigma"]),
        line = dict(color='red', width=2),
        name="Demand band 1 sigma"
        ))
    fig.add_trace(go.Scatter(
        x=observation.index,
        y=((observation["plus_1sigma"])),
        line = dict(color='blue', width=2),
        name="Supplier band 1 sigma"
        ))

    fig.add_trace(go.Scatter(
        x=observation.index,
        y=((observation["minus_2sigma"])),
        line = dict(color='red', width=2, dash='dash'),
        name="Demand band 1 sigma"
        ))
    fig.add_trace(go.Scatter(
        x=observation.index,
        y=((observation["plus_2sigma"])),
        line = dict(color='royalblue', width=2, dash='dash'),
        name="Supplier band 2 sigma"
        ))
    
    fig.add_trace(go.Scatter(
        x=observation.index,
        y=(observation["ref_price"]),
        line = dict(color='orange', width=2, dash='dash'),
        name="Reference Price"
        ))
    
    fig.update_layout(
        title=ticker,
        margin=dict(l=30, r=30, t=50, b=30),
        width=1024,
        height=768
        )
    
    return fig
