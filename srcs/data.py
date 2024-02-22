import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def save_params(data_mean, data_var, params, path):
    theta1 = params[0][0]
    theta0 = params[1][0]
    print(f"\nSaving Params: T^0: {float(theta0[0])}, T^1: {float(theta1[0])}")
    with open(path, 'wb') as f:
            pickle.dump((data_mean, data_var, theta1, theta0), f)

def scale_data(data):
    return  pd.DataFrame((data - data.mean()) / data.std())