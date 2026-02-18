import numpy as np
import pandas as pd
from scipy.stats import norm

from pydantic import BaseModel

class BlackScholes(BaseModel):
    S      : float | None
    K      : float | None
    r      : float | None
    q      : float | None
    sigma  : float | None
    T      : float | None

    def autoExtract(self):
        return tuple(self.model_dump().values())

    def _d1d2(self,t):

        S,K,r,q,sigma,T = self.autoExtract()
            
        log_price = np.log(S/K)
        drift = (r-q)+(sigma**2)/2
        deltaT = T-t
        
        denominator = sigma*np.sqrt(deltaT)

        self.d1 =  (log_price + drift*deltaT)/denominator
        self.d2 = self.d1 - denominator
   
    def Call(self):
        S,K,r,q,sigma,T,t = self.autoExtract()
        self._d1d2(self)

        deltaT = T-t

        Phi_d1 = norm.cdf(self.d1)
        Phi_d2 = norm.cdf(self.d2)

        self.Call_Value = S*Phi_d1 - K*np.exp(-r*deltaT)*Phi_d2
        
        return self.Call_Value
    
    def Put(self):
        S,K,r,q,sigma,T,t = self.autoExtract(self)

        if "Call_Value" == None:
            self.Put_Value = self.Call(t) + (K*np.exp(-r*T)-S*np.exp(-q*T))
        else:
            self.Put_Value = self.Call_Value + (K*np.exp(-r*T)-S*np.exp(-q*T))
                
        return self.Put_Value
            
    def Delta(self):
        self.Delta_value = np.exp(-self.q*self.T)*norm.cdf(self.d1)

        return self.Delta_value
    
    def Vega(self):
        if "Delta_value" == None:
             return self.S*np.sqrt(self.T)*self.Delta()
        else:
            return self.S*np.sqrt(self.T)*self.Delta_Value
        
    def Forward(self):
        Forward= self.S*np.exp((self.r-self.q)*self.T)

        return Forward


    def IV(self,S,K,r,q,sigma,T,t):
        pass

