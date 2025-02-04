import copy

import numpy as np




def compute_cost(X, y, w, b):
    """
        compute cost
        Args:
          X (ndarray (m,n)): Data, m examples with n features
          y (ndarray (m,)) : target values
          w (ndarray (n,)) : model parameters
          b (scalar)       : model parameter

        Returns:
          cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost += (f_wb - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    """
        Computes the gradient for linear regression
        Args:
          X (ndarray (m,n)): Data, m examples with n features
          y (ndarray (m,)) : target values
          w (ndarray (n,)) : model parameters
          b (scalar)       : model parameter

        Returns:
          dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
          dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]
        dj_dw = dj_dw + error * X[i]
        dj_db = dj_db + error
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

def gradient_descent(X, y, w_initial, b_initial, cost_function, gradient_function, alpha, num_iterations):
    """
        Performs batch gradient descent to learn w and b. Updates w and b by taking
        num_iters gradient steps with learning rate alpha

        Args:
          X (ndarray (m,n))   : Data, m examples with n features
          y (ndarray (m,))    : target values
          w_initial (ndarray (n,)) : initial model parameters
          b_initial (scalar)       : initial model parameter
          cost_function       : function to compute cost
          gradient_function   : function to compute the gradient
          alpha (float)       : Learning rate
          num_iterations (int)     : number of iterations to run gradient descent

        Returns:
          w (ndarray (n,)) : Updated values of parameters
          b (scalar)       : Updated value of parameter
          j_history (ndarray (n,)): History of cost
        """
    # An array to store cost J and w's at each iteration primarily for graphing later
    j_history = []
    w = copy.deepcopy(w_initial)  # avoid modifying function param
    b = b_initial
    for i in range(num_iterations):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        j_history.append(cost_function(X, y, w, b)) # Save cost J at each iteration
        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")
    return w, b, j_history
