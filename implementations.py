import numpy as np

# ============================================================
# 1. MEAN SQUARED ERROR - GRADIENT DESCENT
# ============================================================


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""

    w = initial_w.copy()
    n = len(y)

    for _ in range(max_iters):

        predictions = tx @ w

        errors = y - predictions

        gradient = -(tx.T @ errors) / n

        w = w - gamma * gradient

    errors = y - tx @ w
    loss = np.sum(errors**2) / (2 * n)

    return w, loss


# ============================================================
# 2. MEAN SQUARED ERROR - STOCHASTIC GRADIENT DESCENT
# ============================================================


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""

    w = initial_w.copy()
    n = len(y)

    for _ in range(max_iters):

        i = np.random.randint(n)

        xi = tx[i]
        yi = y[i]

        prediction = xi @ w

        error = yi - prediction

        gradient = -xi * error

        w = w - gamma * gradient

    errors = y - tx @ w
    loss = np.sum(errors**2) / (2 * n)

    return w, loss


# ============================================================
# 3. LEAST SQUARES
# ============================================================


def least_squares(y, tx):
    """Calculate least squares using normal equations."""

    xtx = tx.T @ tx

    xty = tx.T @ y

    try:
        w = np.linalg.solve(xtx, xty)

    except np.linalg.LinAlgError:
        # Handle singular matrices
        w = np.linalg.pinv(xtx) @ xty

    errors = y - tx @ w
    loss = np.sum(errors**2) / (2 * len(y))

    return w, loss


# ============================================================
# 4. RIDGE REGRESSION
# ============================================================


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""

    n = len(y)
    d = tx.shape[1]

    xtx = tx.T @ tx

    xty = tx.T @ y

    identity = np.eye(d)

    a = xtx + 2 * n * lambda_ * identity

    try:
        w = np.linalg.solve(a, xty)

    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ xty

    errors = y - tx @ w
    loss = np.sum(errors**2) / (2 * n)

    return w, loss


# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================


def sigmoid(t):
    """Sigmoid function 1 / (1 + exp(-t)), written with tanh to avoid overflow."""

    return 0.5 * (1 + np.tanh(0.5 * t))


def compute_logistic_loss(y, tx, w):
    """Negative log-likelihood averaged over the samples (y in {0, 1})."""

    z = tx @ w

    # np.logaddexp(0, z) computes log(1 + exp(z)) without overflow
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the logistic loss."""

    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1})."""

    w = initial_w.copy()

    for _ in range(max_iters):

        gradient = compute_logistic_gradient(y, tx, w)

        w = w - gamma * gradient

    loss = compute_logistic_loss(y, tx, w)

    return w, loss


# ============================================================
# 6. REGULARIZED LOGISTIC REGRESSION
# ============================================================


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
    (y in {0, 1}, regularization term lambda_ * ||w||^2)."""

    w = initial_w.copy()

    for _ in range(max_iters):

        # Gradient of the penalty lambda_ * ||w||^2 is 2 * lambda_ * w
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w

        w = w - gamma * gradient

    loss = compute_logistic_loss(y, tx, w)

    return w, loss
