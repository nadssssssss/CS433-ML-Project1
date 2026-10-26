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
