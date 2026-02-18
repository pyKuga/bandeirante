import pandas as pd
import numpy as np

from .string_lib import vmatch
import re
from typing import Optional

def volatility_calculation(
    equity_data: pd.DataFrame, 
    equity: str, 
    r: float
) -> Optional[pd.DataFrame]:
    """
    Calcula a curva de volatilidade implícita a partir de dados de opções.
    
    Args:
        equity_data: DataFrame com dados de ações e opções
        equity: Código da ação base
        r: Taxa livre de risco
    
    Returns:
        DataFrame com volatilidade e DeltaT por data de vencimento, ou None se não houver opções
    """
    
    # Separa ativos por tipo usando regex compilado para melhor performance
    pattern_call = re.compile(f'^{equity}[A-L]{{1}}[0-9]{{2,3}}E*$')
    pattern_put = re.compile(f'^{equity}[M-X]{{1}}[0-9]{{2,3}}E*$')
    pattern_equity = re.compile(f'^{equity}[0-9]{{1}}$')
    
    equity_related_list = equity_data["CODNEG"].unique()
    
    # Filtra cada tipo de ativo
    calls_list = [code for code in equity_related_list if pattern_call.match(code)]
    puts_list = [code for code in equity_related_list if pattern_put.match(code)]
    equity_name = [code for code in equity_related_list if pattern_equity.match(code)]
    
    # Valida se existem opções
    if not calls_list and not puts_list:
        return None
    
    if not equity_name:
        raise ValueError(f"Ação base '{equity}' não encontrada nos dados")
    
    # Obtém preço atual da ação
    equity_info = equity_data[equity_data["CODNEG"].isin(equity_name)]
    if equity_info.empty:
        raise ValueError(f"Sem dados para a ação '{equity}'")
    
    current_price = equity_info["PREULT"].iloc[0]
    
    # Filtra e prepara dados das opções
    options_codes = calls_list + puts_list
    equity_options = equity_data[equity_data["CODNEG"].isin(options_codes)].copy()
    
    # Converte datas e calcula DeltaT
    equity_options["datven_time"] = pd.to_datetime(
        equity_options["DATVEN"].astype(str), 
        format="%Y%m%d"
    )
    equity_options["DeltaT"] = (
        equity_options["datven_time"] - pd.to_datetime(equity_options.index)
    ).dt.days #/ 365.0  # Normaliza para anos
    
    # Filtra opções com DeltaT válido
    equity_options = equity_options[equity_options["DeltaT"] > 0]
    
    if equity_options.empty:
        return None
    
    # Calcula forward price
    equity_options["Forward"] = current_price * np.exp(r * equity_options["DeltaT"])
    
    # Identifica calls e filtra opções válidas
    equity_options["isCall"] = equity_options["CODNEG"].isin(calls_list)
    
    # Filtro vetorizado: calls acima do forward, puts abaixo
    mask_valid = (
        (equity_options["isCall"] & (equity_options["PREEXE"] >= equity_options["Forward"])) |
        (~equity_options["isCall"] & (equity_options["PREEXE"] <= equity_options["Forward"]))
    )
    valid_options = equity_options[mask_valid].sort_values(by=["DATVEN", "PREEXE"])
    
    if valid_options.empty:
        return None
    
    # Calcula volatilidade por data de vencimento
    volatility_results = []
    
    for date, group in valid_options.groupby("DATVEN"):
        volatility = _calculate_volatility_for_maturity(group, r)
        if volatility is not None:
            volatility_results.append({
                "Date": pd.to_datetime(str(date), format="%Y%m%d"),
                "Volatility": volatility,
                "DeltaT": group["DeltaT"].iloc[0]
            })
    
    if not volatility_results:
        return None
    
    # Cria DataFrame final
    volatility_curve = pd.DataFrame(volatility_results).set_index("Date").sort_index()
    
    return volatility_curve


def _calculate_volatility_for_maturity(options_group: pd.DataFrame, r: float) -> Optional[float]:
    """
    Calcula volatilidade para um grupo de opções com mesmo vencimento.
    
    Args:
        options_group: DataFrame com opções de mesmo vencimento
        r: Taxa livre de risco
    
    Returns:
        Volatilidade implícita ou None se cálculo falhar
    """
    if len(options_group) < 2:
        return None
    
    # Calcula deltaK (espaçamento entre strikes)
    strikes = options_group["PREEXE"].values
    deltaK = np.zeros(len(strikes))
    
    # Primeiro strike: diferença para o próximo
    deltaK[0] = strikes[1] - strikes[0]
    
    # Strikes intermediários: média das diferenças
    if len(strikes) > 2:
        deltaK[1:-1] = (strikes[2:] - strikes[:-2]) / 2
    
    # Último strike: diferença do anterior
    deltaK[-1] = strikes[-1] - strikes[-2]
    
    # Calcula integral para volatilidade
    prices = options_group["PREMED"].values
    strikes_squared = strikes ** 2
    
    integrals = np.sum((prices * deltaK) / strikes_squared)
    
    # Fator constante
    delta_t = options_group["DeltaT"].iloc[0]
    exp_factor = np.exp(r * delta_t)
    constant_factor = (2 * exp_factor) / delta_t
    
    # Volatilidade
    variance = integrals * constant_factor
    
    if variance <= 0:
        return None
    
    return np.sqrt(variance)


import numpy as np
import pandas as pd
from typing import Optional


def calculate_weighted_volatility(volatility_curve: Optional[pd.DataFrame]) -> float:
    """
    Calcula volatilidade implícita ponderada pelo DeltaT.
    
    Args:
        volatility_curve: DataFrame com colunas 'Volatility' e 'DeltaT'
    
    Returns:
        Volatilidade ponderada ou NaN se inválido
    """
    if volatility_curve is None or volatility_curve.empty:
        return np.nan
    
    # Remove linhas com valores inválidos
    valid_data = volatility_curve.dropna(subset=["Volatility", "DeltaT"])
    
    if valid_data.empty or valid_data["DeltaT"].sum() == 0:
        return np.nan
    
    # Média ponderada: sum(vol_i × deltaT_i) / sum(deltaT_i)
    weights = valid_data["DeltaT"]
    weighted_vol = np.average(valid_data["Volatility"], weights=weights)
    
    return weighted_vol


def build_implied_volatility_timeseries(
    dataset: pd.DataFrame,
    selic: pd.DataFrame,
    equity: str,
    volatility_calc_func: callable
) -> pd.DataFrame:
    """
    Constrói série temporal de volatilidade implícita para uma ação.
    
    Args:
        dataset: DataFrame com dados de mercado (index = datas)
        selic: DataFrame com taxas SELIC (index = datas)
        equity: Código da ação
        volatility_calc_func: Função de cálculo de volatilidade
    
    Returns:
        DataFrame com volatilidade implícita por data
    """
    dates = dataset.index.unique()
    results = []
    
    for date in dates:
        # Obtém dados do dia
        equity_data = dataset.loc[date]
        
        # Obtém taxa SELIC do dia (em decimal)
        try:
            r = 0.01 * selic.loc[date].values[0]
        except (KeyError, IndexError):
            # Se não houver SELIC para o dia, pula
            results.append({"Date": date, "vol": np.nan})
            continue
        
        # Calcula curva de volatilidade
        volatility_curve = volatility_calc_func(equity_data, equity, r)
        
        # Calcula volatilidade ponderada
        weighted_vol = calculate_weighted_volatility(volatility_curve)
        
        results.append({"Date": date, "vol": weighted_vol})
    
    # Cria DataFrame final
    implied_vol = pd.DataFrame(results).set_index("Date").sort_index()
    
    return implied_vol


# Uso (versão compacta para seu caso específico):
def quick_implied_vol_calculation(dataset, selic, equity, volatility_calc_func):
    """Versão simplificada para uso direto."""
    return build_implied_volatility_timeseries(dataset, selic, equity, volatility_calc_func)


# ============== EXEMPLO DE USO ==============

# Versão original otimizada (se preferir manter a estrutura):
def build_implied_vol_optimized(dataset, SELIC, equity, volatility_calc_func):
    """
    Versão otimizada mantendo estrutura similar ao código original.
    """
    dates = dataset.index.unique()
    volatilities = np.empty(len(dates))
    
    for i, date in enumerate(dates):
        equity_data = dataset.loc[date]
        
        try:
            r = 0.01 * SELIC.loc[date].values[0]
            volatility_curve = volatility_calc_func(equity_data, equity, r)
            volatilities[i] = calculate_weighted_volatility(volatility_curve)
        except (KeyError, IndexError, Exception):
            volatilities[i] = np.nan
    
    return pd.DataFrame({"vol": volatilities}, index=dates).sort_index()


# ============== USO ==============
# equity = "BBAS"
# 
# # Opção 1: Versão completa
# implied_vol = build_implied_volatility_timeseries(
#     dataset, SELIC, equity, bd.volatility_calculation
# )
# 
# # Opção 2: Versão otimizada (mais rápida)
# implied_vol = build_implied_vol_optimized(
#     dataset, SELIC, equity, bd.volatility_calculation
# )