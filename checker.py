import pandas as pd
import os
import pickle

def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def check_data(mileage, estimated_price, actual_price):
    deff = abs(estimated_price - actual_price)
    if deff <= 1500:
        print(f"({mileage}) Estimated Price : {estimated_price} | Actual Price : {actual_price} | Difference : {deff} | Success")
    else:
        print(f"({mileage}) Estimated Price : {estimated_price} | Actual Price : {actual_price} | Difference : {deff} | Failure")
    return deff <= 1500

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    model_file = 'model_params.pkl'

    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            loaded_params = pickle.load(f)
        km_mean , price_mean = loaded_params[0]["km"], loaded_params[0]["price"]
        km_std , price_std = loaded_params[1]["km"], loaded_params[1]["price"]
        if (km_mean == 0 or price_mean == 0 or km_std == 0 or price_std == 0):
            print("Error: Model Params are not valid")
        theta0 = float(loaded_params[2][0])
        theta1 = float(loaded_params[3][0])
        print(f"{km_mean}, {price_mean}, {km_std}, {price_std}")
        print(f"Loaded Params : {(theta0)}, {(theta1)}\n")
        pass
    else:
        print("Error: Model Params are not valid")
        exit()

    sum_succes = 0
    for i, row in data.iterrows():
        mileage = (row[0] - km_mean) / km_std
        estimated_price = predict_price(mileage, theta0, theta1)
        estimated_price = (estimated_price * price_std) + price_mean
        estimated_price = round(float(estimated_price), 2)
        sum_succes = sum_succes + check_data(row[0], estimated_price, row[1])
    
    print(f"Success {sum_succes} ove {len(data)} , Success rate : {sum_succes / len(data)}")