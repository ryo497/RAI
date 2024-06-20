from sklearn.metrics import mutual_info_score

def conditional_mutual_information(X, Y, Z, data):
    # 相互情報量の計算
    mi = mutual_info_score(data[X], data[Y])
    
    # 条件付き相互情報量の計算
    cmi = mi - mutual_info_score(data[X], data[Z]) - mutual_info_score(data[Y], data[Z])
    
    return cmi
