# %%
import pandas as pd
import bandeirante as bd
import numpy as np

import datetime as dt

import yfinance as yf

from matplotlib import pyplot as plt

# %%
tickers = ["JURO","HGLG","KFOF","XPML","TRXF","SNID","MXRF","CDII","CRAA","KORE","NDIV","SNAG","EGAF","BOVA"]
tickers = [item + "11.SA" for item in tickers]
#tickers = [,"^BVSP","BTC-USD","BBAS3.SA","PETR3.SA","VALE3.SA"]


end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=5*365)


dataset = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    progress=False
)

dataset.ffill(axis=0,inplace=True)



# %%
import plotly.graph_objects as go

# --- Exemplo de dados
figs = {}

for t in tickers:
    fig = bd.generate_plotly_graph(t,dataset)
    figs[t] = fig

# --- Gerar HTML
html_parts = []

# Dropdown de tickers
dropdown_html = """
<select id="tickerSelect" onchange="showTicker(this.value)">
  {}
</select>
""".format(
    "\n  ".join([f'<option value="{t}">{t}</option>' for t in tickers])
)

# --- Gerar divs com os gráficos
html_parts = []
for i, t in enumerate(tickers):
    display = "block" if i == 0 else "none"
    # Inclui o plotly.js apenas no primeiro gráfico
    fig_html = figs[t].to_html(
        include_plotlyjs=("cdn" if i == 0 else False),
        full_html=True,
        div_id=t,
        config={"responsive": True}
    )
    html_parts.append(f'<div id="{t}_container" style="display:{display}">{fig_html}</div>')

# --- Script JS para alternar os gráficos
script = """
<script>
function showTicker(ticker) {
  const tickers = %s;
  tickers.forEach(t => {
    const div = document.getElementById(t + "_container");
    if (div) div.style.display = (t === ticker) ? "block" : "none";
  });
}
</script>
""" % tickers

# --- HTML final
html_final = f"""
<html>
<head><meta charset="utf-8"></head>
<body>
<h2>Selecione o ticker:</h2>
{dropdown_html}
{''.join(html_parts)}
{script}
</body>
</html>
"""

# --- Salvar em arquivo
with open("../notebook/gráficos/graficos_tickers.html", "w", encoding="utf-8") as f:
    f.write(html_final)



