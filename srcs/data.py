import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def save_params(params, path):
    theta1 = params[0][0]
    theta0 = params[1][0]
    print(f"\nSaving Params: T^0: {float(theta0[0])}, T^1: {float(theta1[0])}")
    with open(path, 'wb') as f:
            pickle.dump(params, f)

def scale_data(data):
    scaler = StandardScaler()  
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)   
