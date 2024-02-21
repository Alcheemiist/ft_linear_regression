import os
import pickle
from sklearn.preprocessing import StandardScaler


def predict_price(mileage, theta0, theta1):
    scaler = StandardScaler()
    y_pred_scaled =  theta0 + (theta1 * mileage)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    return y_pred

if __name__ == '__main__':
    model_file = 'model_params.pkl'
    theta0 = 0
    theta1 = 0

    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            loaded_params = pickle.load(f)

        theta0 = float(loaded_params[0][0])
        theta1 = float(loaded_params[1][0])

        print(f"Loaded Params : {(theta0)}, {(theta1)}\n")
        pass

    print("Given a Milegae of a car, i'll predict the price of the car!!")
    mileage = float(input("Enter the mileage km of the car: "))
    esimated_price = predict_price(mileage, theta0, theta1)
    print(f"The Estimated price : {esimated_price}\n")