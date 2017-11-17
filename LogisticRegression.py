import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propegate(w, b, X, Y):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    grads = {
        'dw' : dw,
        'db' : db
    }

    return grads, cost


def optimaiz(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propegate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i : %f" % (i, cost))

    params = {
        'w' : w,
        'b' : b
    }
    grads = {
        'dw' : dw,
        'db' : db
    }

    return params, grads, costs


def predict(w, b ,X):
    m = X.shape[1]
    Y_predict = np.zeros((1, m))
    w = w.reshape((m, 1))

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        if A[0, i] >= 0.5:
            Y_predict[0, i] = 1
        else:
            Y_predict[0, i] = 0

    return Y_predict


def model(X_train, X_test, Y_train, Y_text, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimaiz(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params['w']
    b = params['b']

    Y_train_predict = predict(w, b, X_train)
    Y_test_predict = predict(w, b, X_test)

    d = {
        'costs' : costs,
        'Y_train_predict' : Y_train_predict,
        'Y_test_predict' : Y_test_predict,
        'dw' : dw,
        'db' : db,
        'learning_rate' : learning_rate,
        'num_iterations' : num_iterations
    }

    return d