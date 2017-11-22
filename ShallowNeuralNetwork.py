import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def layer_size(n_h):
    return n_h


def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]

    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    cost = (1 / m) * np.sum(logprobs, axis = 0)

    cost = np.squeeze(cost)

    return cost


def backward_propagation(X, Y, cache, parameters):
    m = X.shape[1]

    A1 = cache['A1']
    A2 = cache['A2']
    W2 = parameters['W2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims = True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return grads


def update_parameters(parameters, grads, learn_rate = 1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 -= learn_rate * dW1
    b1 -= learn_rate * db1
    W2 -= learn_rate * dW2
    b2 -= learn_rate * db2

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parameters


def model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    parameters = initialize_parameters(X.shape[0], n_h, Y.shape[0])

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(X, Y, cache, parameters)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost = ", (i, cost))

    return parameters


def predict(X, parameters):
    A2, cache = forward_propagation(X, parameters)

    prediction = np.round(A2)

    return prediction


def main():
    parameters = model(X, Y, n_h, num_iterations=10000)

    priedicion = predict(X, parameters)


main()