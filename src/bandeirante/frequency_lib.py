import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sktime.transformations.series.vmd import VmdTransformer
import emd

from scipy.signal import hilbert

def pdEMD(data,span=9):
    emdS = emd.sift.sift(data["close"].ewm(span=span).mean().to_numpy())
    return pd.DataFrame(emdS,index=data.index)

def EMDPlot(emdResult,data):
    n = emdResult.shape[1]
    fig, ax = plt.subplots(n+1,1,figsize=(20,10))


    data["close"].plot(ax=ax[0])#["Adj Close"].plot(ax=ax[0])
    for i in range(1,n+1):
        emdResult[i-1].plot(ax=ax[i])
        #ax[i].set_yticks(indexData.index)



def pdVMD(data,transformer = VmdTransformer(K=7)):
    vmdS = transformer.fit_transform(data.to_numpy())
    return pd.DataFrame(vmdS,index=data.index)

def VMDPlot(vmdResult,data):
    n = vmdResult.shape[1]
    fig, ax = plt.subplots(n+1,1,figsize=(20,10))


    data.plot(ax=ax[0])#["Adj Close"].plot(ax=ax[0])
    for i in range(1,n+1):
        vmdResult[n-i].plot(ax=ax[i])
        #ax[i].set_yticks(indexData.index)
    return fig, ax

def instantFreqAngle(analyticalSignal, w):
    x,y = analyticalSignal.shape
    instantAngle = np.zeros((x,y))

    for i in range(0,analyticalSignal.shape[1]):
        angleAverage = np.convolve(np.angle(analyticalSignal[:,i]), np.ones(w),'same') / w
        instantAngle[:,i] = np.gradient(np.unwrap(angleAverage))
    return instantAngle



def VMDHilbert(vmdResult):
    analyticalSignal = hilbert(vmdResult)
    signalAmplitude = np.abs(analyticalSignal)
    instantFreq = instantFreqAngle(analyticalSignal,3)
    #instantFreq = np.concatenate((np.zeros((1,n)),instantFreq))
    
    return signalAmplitude,instantFreq

def VMDSpectrum(vmdResult,figsize=(20,10)):
    n = vmdResult.shape[1]
    fig,axs = plt.subplots(n,2,figsize=figsize,sharex=True)
    signalAmplitude,instantFreq = VMDHilbert(vmdResult)
    for i in range(0,n):
        axs[i,0].plot(vmdResult.index,signalAmplitude[:,i])
        axs[i,1].plot(vmdResult.index,instantFreq[:,i])
    fig.suptitle("VMD Spectrum")

    return fig, axs