import numpy as np


class Ft_LinearRegression:

    def __init__(self, data, lr, epochs):
        self.data = data
        self.theta0 = 0.0
        self.theta1 = 0.0

        self.weights = 0
        self.biases = 0

        self.params = [self.weights, self.biases]

        self.learning_r = lr
        self.epochs = epochs

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
        # Multiply the gradient by the x values + Divide x by the number of rows in x to avoid updates that are too large
        w_grad = (x.T / x.shape[0]) @ grad
        b_grad = np.mean(grad, axis=0)

        self.params[0] -= w_grad * lr
        self.params[1] -= b_grad * lr
        return params

    def train_model(self, train_x, train_y, valid_x, valid_y):

        self.params = self.init_params(train_x.shape[1])

        sample_rate = 50
        samples = int(self.epochs / sample_rate)

        historical_ws = np.zeros((samples, train_x.shape[1]))
        historical_gradient = np.zeros((samples,))
        historical_index = np.zeros((samples,))
        h_valid_loss = np.zeros((self.epochs,))
        h_index = np.zeros((self.epochs,))

        y = 0
        for i in range(self.epochs):
            predictions = self.forward(self.params, train_x)
            grad = self.mse_grad(train_y, predictions)
            self.params = self.backward(self.params, train_x, self.learning_r, grad)

            if i % sample_rate == 0:
                # Store historical weights for visualization

                index = int(i / sample_rate)
                historical_index[y] = index
                historical_gradient[index] = np.mean(grad)
                historical_ws[index,:] = self.params[0][:,0]

                # Display validation loss
                predictions = self.forward(self.params, valid_x)
                h_valid_loss[i] = self.mse(valid_y, predictions)
                h_index[i] = i
                print(f"Epoch {i} validation loss: {h_valid_loss[i]}")

        return self.params


    # Create two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot the first graph on the first subplot
    # ax1.scatter(historical_index, historical_ws)
    # ax1.scatter(historical_index, historical_gradient)
    # ax1.set_title("Historical Weights vs Gradient")
    # # scatter the second graph on the second subscatter
    # ax2.scatter(h_index, h_valid_loss)
    # ax2.set_title("Validation Loss vs Epoch")

    # fig.savefig("./analysis/data_analysis_train_model.png")

def review(self, ax1, train_x, train_y, params):
    # Plot the first graph on the first subplot
    predictions = self.forward(params, train_x)
    ax1.scatter(train_x, predictions)
    ax1.plot(train_x, predictions, "y+-")
    ax1.scatter(train_x, train_y)
    ax1.set_title("Gradient Descent Linear Regression")