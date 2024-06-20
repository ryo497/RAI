from scipy.special import gammaln
import pandas as pd
import numpy as np

def bayes_factor(X, Y, Z, data, alpha=0.5):
    # クロス集計表の作成
    contingency_table = pd.crosstab(index=[data[z] for z in Z], columns=[data[X], data[Y]])
    r1 = len(data[X].unique())
    r2 = len(data[Y].unique())
    q = len(data.groupby(Z).size())
    # サンプル数の計算
    n_xyz = contingency_table.values
    N_j = np.sum(n_xyz, axis=1)
    # マージナルライクリフッドの計算
    log_marginal_likelihood = np.sum(gammaln(r1 * r2 * alpha) - gammaln(r1 * r2 * alpha + N_j)) + \
    np.sum(gammaln(alpha + n_xyz) - gammaln(alpha))
    return np.exp(log_marginal_likelihood)
