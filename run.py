"""Train the final model on the whole training set and create the AIcrowd
submission file.

Final model: preprocess_data + logistic regression (gradient descent,
gamma = 0.5, 1000 iterations, zero initialization). The decision threshold is
the one that maximizes F1 on the training set, as in the cross-validation of
training.py. Nothing below is random, so every run gives the same file.
"""

import os

import numpy as np

from evaluation import best_threshold, predict_labels
from helpers import create_csv_submission
from implementations import logistic_regression, sigmoid
from preprocessing import preprocess_data

DATA_DIR = "."  # folder that contains x_train.csv, y_train.csv and x_test.csv
OUTPUT = "submission.csv"
GAMMA = 0.5
MAX_ITERS = 1000


def load_csv(name, **kwargs):
    """Load a CSV file of DATA_DIR without its header."""

    path = os.path.join(DATA_DIR, name)
    return np.genfromtxt(path, delimiter=",", skip_header=1, **kwargs)


if __name__ == "__main__":

    # Raw data, Id column included: preprocess_data removes it itself.
    # (helpers.load_csv_data is not used: it already removes the Id column,
    # so preprocess_data would then drop a real feature.)
    print("Loading the data...", flush=True)
    x_train = load_csv("x_train.csv")
    x_test = load_csv("x_test.csv")
    y_data = load_csv("y_train.csv", dtype=int)
    test_ids = x_test[:, 0].astype(int)

    # The labels must be in the same order as the training samples
    assert np.array_equal(y_data[:, 0], x_train[:, 0].astype(int))

    # Labels {-1, 1} -> {0, 1} for logistic regression
    y = (y_data[:, 1] == 1).astype(float)

    x_train_clean, x_test_clean = preprocess_data(x_train, x_test)
    tx_train = np.c_[np.ones(len(x_train_clean)), x_train_clean]
    tx_test = np.c_[np.ones(len(x_test_clean)), x_test_clean]
    print(f"Features (with bias): {tx_train.shape[1]}", flush=True)

    print("Training...", flush=True)
    initial_w = np.zeros(tx_train.shape[1])
    w, loss = logistic_regression(y, tx_train, initial_w, MAX_ITERS, GAMMA)

    threshold, train_f1 = best_threshold(y, sigmoid(tx_train @ w))
    print(f"Train loss={loss:.4f}  threshold={threshold:.2f}  train F1={train_f1:.4f}")

    y_pred = predict_labels(sigmoid(tx_test @ w), threshold)
    print(f"Predicted positives in x_test: {y_pred.mean():.1%}")

    # Predictions {0, 1} -> {-1, 1}, as required by AIcrowd
    create_csv_submission(test_ids, 2 * y_pred - 1, OUTPUT)
    print(f"Saved {len(test_ids)} predictions to {OUTPUT}")
