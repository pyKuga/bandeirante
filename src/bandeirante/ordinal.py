import numpy as np
from itertools import permutations
import math

class OrdinalPatternChecker:
    def __init__(self,n):
        self.pattern_length = n

        self.patterns = np.array(list(permutations(range(n)))) 

        self.n_of_patterns = len(self.patterns)

        self.map = {tuple(p) : i for i,p in enumerate(self.patterns)}

    def check_pattern(self,v):

        rank = np.argsort(np.argsort(v+1e-10*np.random.randn(self.pattern_length)))
        return self.map[tuple(rank)]
    
    def patterns_on_series(self,data):
        return data.rolling(window=self.pattern_length).apply(
            lambda w: self.check_pattern(w),
            raw = True
        )
    
    def entropy(self,v):
        _,D = np.unique(v,return_counts=True)

        D = D/D.sum()

        return -np.dot(D,np.log(D))/np.log(math.factorial(self.pattern_length))
    
    def entropy_on_series(self,data,window_length):
        series_with_patterns = self.patterns_on_series(data)
        return series_with_patterns.rolling(window=window_length, min_periods=0).apply(
            lambda w: self.entropy(w)
        )
