import numpy as np

from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
)

np.random.seed(42)

tx = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=float)

y = np.array([3, 5, 7, 9], dtype=float)

initial_w = np.zeros(2)

w_gd, loss_gd = mean_squared_error_gd(y, tx, initial_w, 10000, 0.01)

print("GD")
print("Weights:", w_gd)
print("Loss:", loss_gd)

w_sgd, loss_sgd = mean_squared_error_sgd(y, tx, initial_w, 10000, 0.01)

print("\nSGD")
print("Weights:", w_sgd)
print("Loss:", loss_sgd)

w_ls, loss_ls = least_squares(y, tx)

print("\nLeast Squares")
print("Weights:", w_ls)
print("Loss:", loss_ls)

w_ridge, loss_ridge = ridge_regression(y, tx, lambda_=0.01)

print("\nRidge Regression")
print("Weights:", w_ridge)
print("Loss:", loss_ridge)
