"""Improvements tried after the model comparison, with 5-fold cross-validation.

The reference is the first final model: preprocess_data with the features
that have more than 80% missing values dropped, then logistic regression
(gamma = 0.5, 1000 iterations). Each variant changes some of its settings. The
folds are the same as in training.py, so every variant is compared fold by
fold with the reference. The best variant (features with up to 99.9% missing
values) is the final model: MAX_MISSING = 0.999 in preprocessing.py.
"""

import numpy as np

import preprocessing
from evaluation import best_threshold, compute_metrics, predict_labels
from implementations import logistic_regression, sigmoid
from training import build_k_indices, load_data

GAMMA = 0.5
MAX_ITERS = 1000
K_FOLD = 5

# Settings of the first final model; each variant changes some of them
REFERENCE = {"max_missing": 0.80, "max_categories": 10, "pos_weight": 1}


def weighted_logistic_regression(y, tx, pos_weight, max_iters, gamma):
    """Logistic regression where a positive sample counts pos_weight times as
    much as a negative one in the loss. Positives have weight 1 and negatives
    1 / pos_weight: with larger weights, gamma = 0.5 makes the loss oscillate."""

    weights = np.where(y == 1, 1.0, 1.0 / pos_weight)

    w = np.zeros(tx.shape[1])
    for _ in range(max_iters):
        gradient = tx.T @ (weights * (sigmoid(tx @ w) - y)) / len(y)
        w = w - gamma * gradient

    return w


def cross_validate(x, y, k_indices, max_missing, max_categories, pos_weight):
    """F1 per fold, mean accuracy and number of features of the pipeline with
    the given settings."""

    # preprocess_data reads these module constants at each call
    preprocessing.MAX_MISSING = max_missing
    preprocessing.MAX_CATEGORIES = max_categories

    f1_scores, accuracies = [], []
    for k in range(K_FOLD):
        test_idx = k_indices[k]
        train_idx = np.concatenate([k_indices[j] for j in range(K_FOLD) if j != k])

        x_tr, x_te = preprocessing.preprocess_data(x[train_idx], x[test_idx])
        tx_tr = np.c_[np.ones(len(x_tr)), x_tr]
        tx_te = np.c_[np.ones(len(x_te)), x_te]
        y_tr, y_te = y[train_idx], y[test_idx]

        if pos_weight == 1:
            initial_w = np.zeros(tx_tr.shape[1])
            w, _ = logistic_regression(y_tr, tx_tr, initial_w, MAX_ITERS, GAMMA)
        else:
            w = weighted_logistic_regression(y_tr, tx_tr, pos_weight, MAX_ITERS, GAMMA)

        # Threshold chosen on the training folds, metrics on the test fold
        threshold, _ = best_threshold(y_tr, sigmoid(tx_tr @ w))
        metrics = compute_metrics(y_te, predict_labels(sigmoid(tx_te @ w), threshold))

        f1_scores.append(metrics["f1"])
        accuracies.append(metrics["accuracy"])
        print(
            f"    fold {k + 1}/{K_FOLD}  threshold={threshold:.2f}  F1={metrics['f1']:.4f}",
            flush=True,
        )

    return np.array(f1_scores), np.mean(accuracies), tx_tr.shape[1]


if __name__ == "__main__":

    x_train, y_train = load_data()
    y = (y_train == 1).astype(float)
    k_indices = build_k_indices(len(y), K_FOLD)  # same folds as training.py

    variants = {
        "Reference (<= 80% missing)": {},
        "Keep features <= 95% missing": {"max_missing": 0.95},
        "Keep features <= 99% missing": {"max_missing": 0.99},
        "Keep features <= 99.9% missing": {"max_missing": 0.999},
        "99% missing + positive weight 2": {"max_missing": 0.99, "pos_weight": 2},
        "99% missing + positive weight 4": {"max_missing": 0.99, "pos_weight": 4},
    }

    reference_f1 = None
    for name, changes in variants.items():
        print(name, flush=True)
        settings = {**REFERENCE, **changes}
        f1, accuracy, n_features = cross_validate(x_train, y, k_indices, **settings)

        line = (
            f"  => features={n_features}  F1={f1.mean():.4f} +- {f1.std():.4f}  "
            f"accuracy={accuracy:.4f}"
        )
        if reference_f1 is None:
            reference_f1 = f1
        else:
            gain = f1 - reference_f1
            line += f"  gain={gain.mean():+.4f} (better in {np.sum(gain > 0)}/{K_FOLD} folds)"
        print(line, flush=True)
