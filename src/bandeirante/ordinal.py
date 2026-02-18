import numpy as np
from itertools import permutations
import math

def OrdinalPatterns(V,n):

    #gera a matriz autoregressiva
    AR = np.array([np.roll(V,i) for i in range(0,n)]).T

    Var =np.flip(AR[n-1:],1)
    
    #executa todas as permutações
    perm = np.array([np.array(p) for p in permutations(range(0,n))]) 
    
    Vl = np.array([]).reshape(0,n)

    #avalia os padrões ordinais do vetor V
    for v in Var:
        U = np.unique(v)
        s,i = (np.sort(U),np.argsort(U))

        mapa = dict(zip(s,i))

        ordenador = np.vectorize(lambda x: mapa[x])
        Vl = np.vstack((Vl,ordenador(v)))

    #gera a classificação dos padrões ordinais, testando padrão por padrão em cada array
    classificacao = np.zeros((Vl.shape[0]))

    for i in range(0,len(perm)):
        loc = np.sum(perm[i] == Vl,axis=1)==n
        classificacao[loc] = i
    
    return classificacao

def PatternsEntropy(patterns,n,m):

    H = np.ones(patterns.shape[0]+n-1)

    for i in range(0,patterns.shape[0]-m):

        window = patterns[i:i+m]

        _,D = np.unique(window,return_counts=True)

        D = D/D.sum()

        H[i+n-1+m] =  - np.dot(D,np.log(D))/np.log(math.factorial(n))
    
    H[0:n-1+m] = np.nan

    return H