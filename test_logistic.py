"""Check logistic_regression and reg_logistic_regression against the
expected values of the course's official grading tests."""

import numpy as np

from implementations import logistic_regression, reg_logistic_regression

# Same small example as the official tests (labels in {0, 1})
y = np.array([0.0, 1.0, 1.0])
tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
initial_w = np.array([0.5, 1.0])
MAX_ITERS = 2
GAMMA = 0.1


def check(name, w, loss, expected_w, expected_loss):
    assert w.shape == (2,), f"{name}: w must be a 1D vector, got shape {w.shape}"
    assert np.ndim(loss) == 0, f"{name}: loss must be a scalar"
    np.testing.assert_allclose(w, expected_w, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-4, atol=1e-8)
    print(f"{name}: OK  w={w}  loss={loss:.6f}")


w, loss = logistic_regression(y, tx, initial_w, MAX_ITERS, GAMMA)
check("logistic_regression", w, loss, [0.378561, 0.801131], 1.348358)

w, loss = reg_logistic_regression(y, tx, 1.0, initial_w, MAX_ITERS, GAMMA)
check("reg_logistic_regression", w, loss, [0.216062, 0.467747], 0.972165)

# The functions must not modify initial_w (they work on a copy)
np.testing.assert_array_equal(initial_w, [0.5, 1.0])
print("initial_w unchanged: OK")
