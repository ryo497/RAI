import pandas as pd
import numpy as np
from scipy.stats import chi2


def g2_test(X, Y, Z, data):
    # クロス集計表の作成
    contingency_table = pd.crosstab(data[X], [data[Y], data[Z]])
    
    # G²統計量の計算
    observed = contingency_table.values
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    g2 = 2 * np.sum(observed * np.log(observed / expected))
    
    # p値の計算
    dof = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)
    p = 1 - chi2.cdf(g2, dof)
    
    return g2, p
