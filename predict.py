import os
import pickle
from sklearn.preprocessing import StandardScaler


def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

if __name__ == '__main__':
    model_file = 'model_params.pkl'
    theta0 = 0
    theta1 = 0

    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            loaded_params = pickle.load(f)

        # print(loaded_params)
        # print("km: " , loaded_params[0]["km"])

        km_mean , price_mean = loaded_params[0]["km"], loaded_params[0]["price"]
        km_std , price_std = loaded_params[1]["km"], loaded_params[1]["price"]
        if (km_mean == 0 or price_mean == 0 or km_std == 0 or price_std == 0):
            print("Error: Model Params are not valid")
        theta0 = float(loaded_params[2][0])
        theta1 = float(loaded_params[3][0])
        print(f"{km_mean}, {price_mean}, {km_std}, {price_std}")
        print(f"Loaded Params : {(theta0)}, {(theta1)}\n")


        # exit()
        # theta0 = float(loaded_params[0][0])
        # theta1 = float(loaded_params[1][0])

        # print(f"Loaded Params : {(theta0)}, {(theta1)}\n")
        pass

    print("Given a Milegae of a car, i'll predict the price of the car!!")
    mileage = float(input("Enter the mileage km of the car: "))

    mileage = (mileage - km_mean) / km_std

    esimated_price = predict_price(mileage, theta0, theta1)

    esimated_price = (esimated_price * price_std) + price_mean
    esimated_price = round(float(esimated_price), 2)

    print(f"The Estimated price : {esimated_price}\n")