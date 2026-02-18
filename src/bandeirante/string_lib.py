import numpy as np
import pandas as pd
import re

def letra_para_numero(letra):
    return ord(letra.upper()) - ord('A') + 1

def numero_para_letra(numero):
    return chr(numero-1 + ord('A')), chr(numero-1 + ord('M'))
    
def terceira_sexta(mes=None, ano=None): #ChatGPT
    hoje = pd.Timestamp.today()
    ano = ano or hoje.year
    mes = mes or hoje.month
    
    dias = pd.date_range(start=f"{ano}-{mes:02d}-01", end=f"{ano}-{mes:02d}-28", freq='W-FRI')
    
    return dias[2]  # terceira sexta-feira

def vmatch(exp):
    return np.vectorize(lambda x: bool(re.compile(exp).match(x)))