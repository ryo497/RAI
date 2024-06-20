import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling


# サンプルデータの読み込み
def load_data(type):
    model = get_example_model(type)
    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=1000)
    print(data.head())
    return data


