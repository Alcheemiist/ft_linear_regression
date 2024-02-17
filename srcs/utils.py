import numpy as np
import pandas as pd

def read_data_source(input_file_path):
    data = pd.read_csv(input_file_path)
    data = data.ffill()
    df = pd.DataFrame(data.describe())
    df.to_csv("./analysis/data_analysis_describe.csv")
    dt = data.plot.scatter("km", "price")
    dt.get_figure().savefig("./analysis/data_analysis_scatter.png")
    return data
    
def split_data(data_scaled):
    PREDICTORS = ["km"]
    TARGET = "price"
    np.random.seed(0)
    split_data_scaled = np.split(data_scaled, [ int(.85 * len(data_scaled))])
    (train_x, train_y), (valid_x, valid_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in split_data_scaled]
    return  (train_x, train_y), (valid_x, valid_y)

def predict_price(mileage, theta0, theta1): 
    return theta0 + (theta1 * mileage)

def get_estimated_price(mileage, theta0, theta1): 
    return  predict_price(mileage, theta0, theta1)