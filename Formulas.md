
# Formulas

## Linear Regression

- tmpθ0 and tmpθ1 are temporary variables that store the updated values for the parameters of the linear regression model. In the context of linear regression,
- θ0 is the y-intercept and θ1 is the slope of the line.
- learningRate is a hyperparameter that determines the step size at each iteration while moving toward a minimum of a loss function. It's multiplied with the gradient to determine the next position.
- 1/m is a normalization factor, where m is the number of training examples. This is used to average the total error of the model over all training examples.
- The sum from i=0 to m-1 is calculating the total error of the model over all training examples.
- estimatePrice(mileage[i]) - price[i] is the error of the model on a single training example. It's the difference between the model's prediction and the actual value.
- In tmpθ1, (estimatePrice(mileage[i]) - price[i]) * mileage[i] is the derivative of the cost function with respect to θ1. It's used to update the value of θ1.

- The purpose of these calculations is to iteratively adjust the parameters θ0 and θ1 in the direction that minimizes the cost function, which measures the difference between the model's predictions and the actual values. This is the core of the gradient descent algorithm.

## the gradient descent algorithm for linear regression

**Slope** : tmp1 = ((x.T / x.shape[0]) @ grad) * lr

**Intercept** : tmp0 = np.mean(grad, axis=0) * lr

- x.T is the transpose of the matrix x. Transposing a matrix involves flipping it over its diagonal, which switches its row and column indices. This is necessary here because the shapes of x and grad must align for the matrix multiplication operation.
- x.shape[0] returns the number of rows in x, which is equivalent to the number of training examples. Dividing x.T by x.shape[0] normalizes the data, ensuring that the gradient descent algorithm works correctly even if the number of training examples changes.
- @ is the matrix multiplication operator in Python. It multiplies x.T / x.shape[0] by grad. This operation is a key part of the gradient descent algorithm, as it calculates the gradient of the cost function with respect to the parameters of the model.
- The result, tmp1, is a vector that represents the direction in which the parameters of the model should be adjusted to minimize the cost function.