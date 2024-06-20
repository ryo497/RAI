from scipy.stats import chi2_contingency
import pandas as pd


def chi2_test(X, Y, Z, data):
    # クロス集計表の作成 
    contingency_table = pd.crosstab(index=[data[z] for z in Z], columns=[data[X], data[Y]])
    # χ²検定の実行
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p
