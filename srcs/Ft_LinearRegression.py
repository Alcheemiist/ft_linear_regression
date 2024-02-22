import numpy as np
import matplotlib.pyplot as plt

class Ft_LinearRegression:

    def __init__(self, data, lr, epochs, sr):
        self.data = data
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.weights = 0
        self.biases = 0
        self.params = [self.weights, self.biases]
        self.learning_r = lr
        self.epochs = epochs
        self.sample_rate = sr
        self.h_valid_loss = np.zeros((self.epochs)) 
        self.h_index = np.zeros((self.epochs))
 
    def init_params(self, predictors):
        np.random.seed(0)
        self.weights = np.random.rand(predictors, 1)
        self.biases = np.ones((1, 1))
        return  [self.weights, self.biases]

    def forward(self, params, x):
        self.weights, self.biases = self.params
        prediction = x @ self.weights + self.biases
        return prediction

    def mse(self, actual, predicted):
        #  mean squared error
        return np.mean((actual - predicted) ** 2)

    def mse_grad(self, actual, predicted):
        # The derivative of mean squared error
        return predicted - actual

    def backward(self, params, x, lr, grad):
        tmp1 = (x.T / x.shape[0]) @ grad
        tmp0 = np.mean(grad, axis=0)
        
        self.params[0] -= tmp1 * lr
        self.params[1] -= tmp0 * lr
        return params

    def train_model(self, train_x, train_y, valid_x, valid_y):
        self.params = self.init_params(train_x.shape[1])
        samples = int(self.epochs / self.sample_rate)
        historical_ws = np.zeros((samples, train_x.shape[1]))
        historical_gradient = np.zeros((samples,))
        historical_index = np.zeros((samples,))
        self.h_valid_loss = np.zeros((samples ,))
        self.h_index = np.zeros((samples ,))

        y = 0
        for i in range(self.epochs):
            predictions = self.forward(self.params, train_x)
            grad = self.mse_grad(train_y, predictions)
            self.params = self.backward(self.params, train_x, self.learning_r, grad)
            if i % self.sample_rate == 0:
                index = int(i / self.sample_rate)
                historical_index[y] = index
                historical_gradient[y] = np.mean(grad)
                historical_ws[y,:] = self.params[0][:,0]
                predictions = self.forward(self.params, valid_x)
                self.h_valid_loss[y] = self.mse(valid_y, predictions)
                self.h_index[y] = i
                print(f"Epoch {i} validation loss: {self.h_valid_loss[y]}")
                y += 1
        return self.params

    def metrics(self, params, train_x, train_y):
        fig, ax1 = plt.subplots()
        predictions = self.forward(params, train_x)
        y_pred = predictions
        ax1.set_title("Gradient Descent Linear Regression")
        ax1.scatter(train_x, train_y)
        ax1.scatter(train_x, y_pred)
        ax1.plot(train_x, y_pred, "y+-")
        ax1.set_xlabel("Km")
        ax1.set_ylabel("Price")
        ax1.grid()
        ax1.legend(["Actual", "Predictions"])

        fig.savefig("./analysis/data_analysis_train.png")

        ax3 = plt.figure().add_subplot(111)
        # print(list(self.h_valid_loss))
        ax3.plot(self.h_index, self.h_valid_loss, "r+-")
        ax3.set_title("Validation Loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.grid()
        plt.show()

def review(self, ax1, train_x, train_y, params):
    predictions = self.forward(params, train_x)
    ax1.scatter(train_x, predictions)
    ax1.plot(train_x, predictions, "y+-")
    ax1.scatter(train_x, train_y)
    ax1.set_title("Gradient Descent Linear Regression")