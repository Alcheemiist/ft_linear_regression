import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from sklearn.metrics import r2_score

def read_data_source(input_file_path):
    # Read in the data
    data = pd.read_csv(input_file_path)
    data = data.ffill()

    df = pd.DataFrame(data.describe())
    df.to_csv("./analysis/data_analysis_describe.csv")
    dt = data.plot.scatter("km", "price")
    dt.get_figure().savefig("./analysis/data_analysis_scatter.png")
    return data

def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def get_estimated_price(mileage, theta0, theta1):
    esimated_price = predict_price(mileage, theta0, theta1)
    return esimated_price

def SK_linear_regression(data):
    # Fit a linear regression model
    lr = LinearRegression()
    lr.fit(data[["km"]], data["price"])
    # Plot our data points and the regression line
    data_plot= data.plot.scatter("km", "price", color="blue")
    plt.plot(data["km"], lr.predict(data[["km"]]), color="red")
    data_plot.get_figure().savefig("./analysis/data_analysis_Sklearn_LR_scatter.png")
    return lr

def init_params(predictors):
    # Initialize model parameters
    # k is a scaling factor that we use to reduce the weights and biases initially
    k = math.sqrt(1 / predictors)
    # We set a random seed so if we re-run this code, we get the same results
    np.random.seed(0)
    weights = np.random.rand(predictors, 1)
    biases = np.ones((1, 1))
    return  [weights, biases]

def forward(params, x):
    weights, biases = params
    # Multiply x values by w values with matrix multiplication, then add b
    prediction = x @ weights + biases
    return prediction

def mse(actual, predicted):
    # Calculate mean squared error
    return np.mean((actual - predicted) ** 2)

def mse_grad(actual, predicted):
    # The derivative of mean squared error
    return predicted - actual

def backward(params, x, lr, grad):
    # Multiply the gradient by the x values
    # Divide x by the number of rows in x to avoid updates that are too large
    w_grad = (x.T / x.shape[0]) @ grad
    b_grad = np.mean(grad, axis=0)


    params[0] -= w_grad * lr
    params[1] -= b_grad * lr

    return params

def train_model(train_x, train_y, valid_x, valid_y):
    learning_r = 1e-5
    epochs = 300000
    params = init_params(train_x.shape[1])

    sample_rate = 100
    samples = int(epochs / sample_rate)

    historical_ws = np.zeros((samples, train_x.shape[1]))
    historical_gradient = np.zeros((samples,))
    historical_index = np.zeros((samples,))
    h_valid_loss = np.zeros((epochs,))
    h_index = np.zeros((epochs,))

    y = 0
    for i in range(epochs):
        predictions = forward(params, train_x)
        grad = mse_grad(train_y, predictions)
        params = backward(params, train_x, learning_r, grad)

        # Store historical weights for visualization
        if i % sample_rate == 0:
            index = int(i / sample_rate)
            historical_index[y] = index
            historical_gradient[index] = np.mean(grad)
            historical_ws[index,:] = params[0][:,0]
            y += 1

        # Display validation loss
        if i % sample_rate == 0:
            predictions = forward(params, valid_x)
            h_valid_loss[i] = mse(valid_y, predictions)
            h_index[i] = i
            print(f"Epoch {i} validation loss: {h_valid_loss[i]}")

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot the first graph on the first subplot
    ax1.scatter(historical_index, historical_ws)
    ax1.scatter(historical_index, historical_gradient)
    ax1.set_title("Historical Weights vs Gradient")
    # scatter the second graph on the second subscatter
    ax2.scatter(h_index, h_valid_loss)
    ax2.set_title("Validation Loss vs Epoch")

    fig.savefig("./analysis/data_analysis_train_model.png")

    theta1 = params[0]
    theta0 = params[1]
    print(f" Params : {float(theta0)}, {float(theta1)}")

    # Save the model parameters
    with open('./model_params.pkl', 'wb') as f:
        pickle.dump(params, f)
    return params

def review_gd(ax1, train_x, train_y, params):
    # Plot the first graph on the first subplot
    predictions = forward(params, train_x)
    ax1.scatter(train_x, predictions)
    ax1.plot(train_x, predictions, "y+-")
    ax1.scatter(train_x, train_y)
    ax1.set_title("Gradient Descent Linear Regression")

def review_lr(ax2, data, lr):
    # Plot the second graph on the second subplot
    ax2.scatter(data["km"], data["price"])
    ax2.plot(data["km"], lr.predict(data[["km"]]), "r+-", color="green")
    ax2.set_title("Sklearn Linear Regression")

if __name__ == '__main__':
    theta0 = 0
    theta1 = 0

    if False:
        # Load theta0 and theta1 from a file
        pass

    # print("Given a Milegae of a car, i'll predict the price of the car!!")
    # mileage = float(input("Enter the mileage km of the car: "))
    
    data = read_data_source("./data.csv")
    # 1. Check for Missing Values
    missing_values = data.isnull().sum()
    # 2. Feature Scaling
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # 3. Data Splitting
    X = data_scaled[['km']]  # Feature matrix
    y = data_scaled['price']  # Target vector
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = SK_linear_regression(data)

    PREDICTORS = ["km"]
    TARGET = "price"
    # Ensure we get the same split every time
    np.random.seed(0)
    split_data_scaled = np.split(data_scaled, [int(.7 * len(data_scaled)), int(.85 * len(data_scaled))])
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in split_data_scaled]

    params = train_model(train_x, train_y, valid_x, valid_y)

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    review_gd(ax1, train_x, train_y, params)
    review_lr(ax2, data, lr)
    fig.savefig("./analysis/data_analysis_review.png")

    predictions = forward(params, X_test)
    Error_value = np.mean(np.abs(y_test - predictions))
    print(f"Error = {Error_value} / Error(%) = {Error_value * 100} %")

    y_pred = forward(params, X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"Accuracy = {r2} / Accuracy(%) = {r2 * 100} %")
    exit()
    esimated_price = get_estimated_price(mileage, theta0, theta1)
    print(f"The estimated price of the car is: {esimated_price} km of mileage: {mileage}")