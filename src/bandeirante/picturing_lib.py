from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import pandas as pd

from matplotlib import pyplot as plt
import numpy as np

from matplotlib import axes

import plotly.graph_objects as go
import pandas as pd


def GaussianMixturePlot(data,gmm,strings,figsize=(7,5)):
    model = gmm
    means = model.means_.flatten()
    stds = np.sqrt(model.covariances_).flatten()
    weights = model.weights_

    # Criando um range de valores para plotar as distribuições
    x = np.linspace(np.min(data), np.max(data), 1000)


    fig = plt.figure(figsize=figsize)
    # Plotando histograma dos dados originais
    plt.hist(data, bins=100, density=True, alpha=0.5, label=strings[0])
    #plt.xlim(0,1)

    # Plotando cada gaussiana individualmente
    for i in range(model.means_.shape[0]):
        plt.plot(x, weights[i] * norm.pdf(x, means[i], stds[i]), label=f"{strings[1]} {i+1}")

    # Plotando a soma das gaussianas
    #pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    #plt.plot(x, pdf, label="Soma das Gaussianas", color="blue", linestyle="dashed")

    plt.legend()
    plt.title(strings[2])
    plt.xlabel(strings[3])
    plt.ylabel(strings[4])
    #path = "../imagens_gerais/gmm_"+data.name+".jpg"
    #plt.savefig(path)
    plt.show()


    return gmm, fig

def add_state_background(
        finData        : pd.DataFrame,
        Headers         : list,
        State           : str,
        n               : int,
        ax              : axes.Axes,
        chosenPallete   = 'Oranges'
        ):
    
    '''
        Fills the time-series with an transparent mask of each color representing one state. 
    '''

    cmap = plt.get_cmap(chosenPallete, n)
    for state in range(0,n):
            color = cmap(state)  # Pega uma cor automática para cada estado
            ax.fill_between(finData.index,np.min(finData[Headers]), np.max(finData[Headers]), where=(finData[State] == state), 
                            color=color, alpha=0.3, label=f"State {state}")
            ax.legend(loc='upper left',bbox_to_anchor=(1, 1),fontsize=20)


def OverFill_Plotly(
        finData: pd.DataFrame,
        Headers: list,
        State: str,
        n: int,
        chosenPalette='Oranges'
    ):
    """
    Fills the time-series with transparent colored masks for each state.
    """

    fig = go.Figure()
    # Plota as séries principais
    for h in Headers:
        fig.add_trace(go.Scatter(
            x=finData.index,
            y=finData[h],
            mode='lines',
            name=h,
            line=dict(width=2)
        ))

    # Paleta de cores equivalente
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap(chosenPalette, n)

    # Gera máscaras de regiões por estado
    y_min, y_max = finData[Headers].min().min(), finData[Headers].max().max()
    states = finData[State].values
    x = finData.index

    start_idx = 0
    current_state = states[0]

    for i in range(1, len(states)):
        if states[i] != current_state or i == len(states) - 1:
            # Corrige último intervalo
            end_idx = i if i < len(states) - 1 else i + 1
            color = mcolors.to_hex(cmap(current_state))
            fig.add_vrect(
                x0=x[start_idx],
                x1=x[end_idx-1],
                fillcolor=color,
                opacity=0.3,
                line_width=0,
                annotation_text=f"State {current_state}",
                annotation_position="top left",
            )
            start_idx = i
            current_state = states[i]

    fig.update_layout(
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=1.05,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=60, r=200, t=60, b=60)
    )

    return fig

def transformation_components_plot(
        model       ,   #sklearn.transformation -> but this is not a class
        Headers     :   list, 
        listOfNames =   ["PCA - Measures and Variance Components", "Correlation","PCA Components", "Original Data Features"],
        figsize     =   (10,10),
        textsize    =   10,
        titlesize   =   16,
        ticksize    =   10,
        labelsize    =   10,
        cmap         =  "coolwarm",
        textcolor       =   "black"
        ): #-> Tuple[fig, axs]
    
    '''
        Plot the associated matrix from a transformation model. 

        Example:

        X = USV^t -> U is ploted
    '''

    transposedComponents = model.components_.T

    plt.figure(figsize=figsize)

    plt.imshow(transposedComponents, interpolation='nearest',aspect='auto',cmap=cmap)

    for i in range(transposedComponents.shape[0]):
        for j in range(transposedComponents.shape[1]):
            plt.text(j, i, f'{transposedComponents[i, j]:.2f}', ha='center', va='center', color=textcolor,fontsize=textsize)

    
    plt.yticks(ticks=range(0,len(Headers)),labels=Headers,fontsize=ticksize)
    plt.xticks(ticks=range(0,len(Headers)),labels=range(0,len(Headers)),fontsize=ticksize)
    
    
    plt.title(listOfNames[0],fontsize=titlesize)
    plt.colorbar(label=listOfNames[1])
    plt.xlabel(listOfNames[2],fontsize=labelsize)
    plt.ylabel(listOfNames[3],fontsize=labelsize)
    plt.tight_layout()
    return plt.gcf(), plt.gca()