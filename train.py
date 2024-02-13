import os
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta0 = 0
        self.theta1 = 0

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.iterations):
            theta0_gradient = (1/n) * sum([self.theta0 + self.theta1 * X[i] - y[i] for i in range(n)])
            theta1_gradient = (1/n) * sum([(self.theta0 + self.theta1 * X[i] - y[i]) * X[i] for i in range(n)])
            self.theta0 = self.theta0 - (self.learning_rate * theta0_gradient)
            self.theta1 = self.theta1 - (self.learning_rate * theta1_gradient)

    def predict(self, X):
        return [self.theta0 + (self.theta1 * X[i]) for i in range(len(X))]

    def get_params(self):
        return self.theta0, self.theta1

def read_data_source(input_file_path):
    # Read in the data
    data = pd.read_csv(input_file_path)
    data = data.ffill()

    data.describe()
    data.head(5)
    # data.plot.scatter("km", "price")

def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def get_estimated_price(mileage, theta0, theta1):
    esimated_price = predict_price(mileage, theta0, theta1)
    return esimated_price

if __name__ == '__main__':
    theta0 = 0
    theta1 = 0

    if False:
        # Load theta0 and theta1 from a file
        pass

    print("Given a Milegae of a car, i'll predict the price of the car!!")
    mileage = float(input("Enter the mileage km of the car: "))
    
    read_data_source("./data.csv")

    exit()
    esimated_price = get_estimated_price(mileage, theta0, theta1)
    print(f"The estimated price of the car is: {esimated_price} km of mileage: {mileage}")