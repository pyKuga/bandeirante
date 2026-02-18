import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def generate_order_flow(
        dataset: pd.DataFrame, 
        state_col: str, 
        price_col: str = "PREULT",
        short=True
        ) -> np.ndarray:
    """
    Define o fluxo (quando entrar comprado) baseado no retorno médio em cada estado.
    Retorna array binário (0 = fora, 1 = comprado).
    """
    returns = dataset[price_col].pct_change()
    state_of_pos_return = returns.loc[dataset[state_col] == 1].mean() > 0
    flow = np.where(
        dataset[state_col] == state_of_pos_return, 
        1, 
        short*(-1)
        )

    # shift para evitar look-ahead bias
    return np.roll(flow, 1)

def compute_positions(flow: np.ndarray) -> np.ndarray:
    """Identifica pontos de mudança de posição (entrada/saída)."""
    position_change = np.roll(flow, 1) != flow
    position_change[0] = True
    return position_change


def simulate_strategy(
        dataset: pd.DataFrame, 
        flow: np.ndarray, 
        capital_inicial: float = 1000,
        corretagemTx: float = -5, 
        txShort: float = -0.0071, 
        IR: float = -0.15,
        price_col = "PREABE"
        ) -> pd.Series:
    """
    Simula a curva de capital da estratégia.
    Aplica custos de corretagem, taxa de short e imposto de renda.
    """

    pctR = dataset[price_col].pct_change().fillna(0)
    finalResult = (pctR * flow + 1).cumprod() * capital_inicial
    finalResult.iloc[0] = capital_inicial

    positionChange = compute_positions(flow)

    # custos
    corretagem = positionChange * corretagemTx
    taxa_short = (flow == -1) * (txShort * finalResult)  # se houver short - isto pode não estar certo
    operationTimeSeries = finalResult + corretagem + taxa_short

    # imposto de renda sobre operações lucrativas
    cashFlow = (finalResult.loc[positionChange] - finalResult.loc[positionChange].shift(1)).fillna(0)
    IRFlow = cashFlow.loc[cashFlow > 0] * IR

    retorno_absoluto = (operationTimeSeries.iloc[-1] + IRFlow.sum()) / capital_inicial

    return operationTimeSeries, retorno_absoluto


def plot_strategy_results(dataset: pd.DataFrame, operationTimeSeries: pd.Series, capital_inicial: float = 1000,price_col="PREABE"):
    
    """
    Plota a curva da estratégia vs. Buy and Hold e marca linha de separação.
    """
    plt.figure(figsize=(20, 10))
    (operationTimeSeries / capital_inicial).plot(label="Modelo Computacional")
    (dataset[price_col].pct_change().add(1).cumprod()).plot(label="Buy and Hold")


    plt.legend()
    plt.title("Comparação entre Retornos")
    plt.show()
    

def train_test_split(X,y,fraction):
    total_entries = X.shape[0]

    limit = int(total_entries*fraction)

    X_train = X.iloc[:limit]
    y_train = y.iloc[:limit]

    X_test = X.iloc[limit:]
    y_test = y.iloc[limit:]

    print("eventos totais: " + str(total_entries))
    print("eventos treino: " + str(limit))

    return X_train, X_test, y_train, y_test