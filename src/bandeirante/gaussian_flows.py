import pandas as pd
from sklearn.mixture import GaussianMixture

def GaussianMixtureFit(data,n_components,seed=19971215):
    if type(data) == pd.Series:
        reshapedData = data.to_numpy().reshape(-1,1)
    else:
        reshapedData = data.to_numpy()
    gmm = GaussianMixture(n_components=n_components, random_state=seed,covariance_type="full")
    gmm.fit(reshapedData)
    print("GMM BIC: " + str(gmm.bic(reshapedData)))

    return gmm