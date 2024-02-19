import pandas as pd
import os
import pickle

def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def check_data(estimated_price, actual_price):
    deff = abs(estimated_price - actual_price)
    if deff < 1000:
        print(f"Estimated Price : {estimated_price} | Actual Price : {actual_price} | Difference : {deff} | Success")
    else:
        print(f"Estimated Price : {estimated_price} | Actual Price : {actual_price} | Difference : {deff} | Failure")
    return deff < 1000

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    
    model_file = 'model_params.pkl'
    theta0 = 0
    theta1 = 0
    sum_succes = 0

    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            loaded_params = pickle.load(f)

        theta0 = float(loaded_params[0])
        theta1 = float(loaded_params[1])

        print(f"Loaded Params : {(theta0)}, {(theta1)}\n")
        pass

    for i, row in data.iterrows():
        estimated_price = predict_price(row[0], theta0, theta1)
        sum_succes = sum_succes + check_data(estimated_price, row[1])
    
    print(f"Success {sum_succes} ove {len(data)} , Success rate : {sum_succes / len(data)}")