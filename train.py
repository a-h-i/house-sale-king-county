import torch
from tqdm import notebook


"""

w := w - alpha * dl/dw

dl/dw = dl/dy_ * dy_/dw

dy_/dw = X
dl/dy_ = y_ / m

y_ = X.W + b
L = (y_ - y)^2 / 2m

dl/dw = X * (y_h - y) / m

dl/db = dl/dy_ * dy_/db
dy_/db = 1
"""


def gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
    num_iterations: torch.Tensor,
    batch_size: int,
    w_initial: torch.Tensor = None,
    b_initial: torch.Tensor = None,
):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (Tensor (m,n))   : Data, m examples with n features
      y (Tensor (m,1))    : target values
      w_initial (Tensor (n,1)) : initial model parameters
      b_initial (scalar)       : initial model parameter
      alpha (float)       : Learning rate
      num_iterations (int)     : number of iterations to run gradient descent

    Returns:
      w (Tensor (n,1)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      j_history (Tensor (n,1)): History of cost
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    w = w_initial or torch.rand((X.shape[1], 1), device=X.device)
    b = b_initial or torch.rand(1, device=X.device)
    j_history = []
    for _ in (prog_bar := notebook.trange(num_iterations, desc="learning")):
        epoch_loss = []
        for i in range(0, len(X), batch_size):
            Xmini = X[i:i+batch_size]
            ymini = y[i:i+batch_size]
            m = len(Xmini)
            prediction = Xmini @ w + b
            loss = torch.square(prediction - ymini).sum() / (m * 2)
            dj_dy_ = (prediction - ymini) / m
            dj_dw = Xmini.T @ dj_dy_
            dj_db = dj_dy_.sum()
            w -= alpha * dj_dw
            b -= alpha * dj_db
            epoch_loss.append(loss.item())
        j_history.append(sum(epoch_loss) / len(epoch_loss))  # Save cost J at each iteration
        prog_bar.set_description(f"Loss: {j_history[-1]:8.2f}")
    return w, b, j_history
