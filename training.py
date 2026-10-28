import os

import numpy as np

from evaluation import best_threshold, compute_metrics, predict_labels
from implementations import (
    least_squares,
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)
from preprocessing import preprocess_data


def load_data():
    """Load x_train and the labels. The first run reads the CSV files (slow)
    and saves them as .npy files, which later runs load in a few seconds."""

    if os.path.exists("x_train.npy") and os.path.exists("y_train.npy"):
        return np.load("x_train.npy"), np.load("y_train.npy")

    x_train = np.genfromtxt("x_train.csv", delimiter=",", skip_header=1)
    y_train = np.genfromtxt("y_train.csv", delimiter=",", skip_header=1, usecols=1)
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)

    return x_train, y_train


def split_data(x, y, ratio=0.8, seed=1):
    """Randomly split the data: `ratio` of the samples for training,
    the rest for validation. The seed makes the split reproducible."""

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    cut = int(ratio * len(y))
    tr, val = indices[:cut], indices[cut:]

    return x[tr], x[val], y[tr], y[val]


def validation_f1(w, tx_val, y_val):
    """Best threshold and the F1 score it gives on the validation set."""

    probabilities = sigmoid(tx_val @ w)

    return best_threshold(y_val, probabilities)


def print_metrics(name, y_val, scores):
    """Print the validation metrics of a model, at threshold 0.5 and at the
    threshold that maximizes F1."""

    f1_at_05 = compute_metrics(y_val, predict_labels(scores, 0.5))["f1"]
    threshold, _ = best_threshold(y_val, scores)
    m = compute_metrics(y_val, predict_labels(scores, threshold))

    print(
        f"  {name:30s} F1@0.5={f1_at_05:.4f}  threshold={threshold:.2f}  F1={m['f1']:.4f}  "
        f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  accuracy={m['accuracy']:.4f}"
    )

def logistic_scores(tx, w):
    """Scores of a logistic model: predicted probabilities."""

    return sigmoid(tx @ w)


def linear_scores(tx, w):
    """Scores of a linear regression model: tx @ w."""

    return tx @ w


def build_k_indices(n, k_fold, seed=1):
    """Shuffle the indices 0..n-1 and cut them into k_fold groups of
    (almost) equal size. Each group is used once as the test fold."""

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    return np.array_split(indices, k_fold)


def cross_validation(x, y, models, k_fold=5, seed=1):
    """k-fold cross-validation of several models on the same folds.

    For each fold, preprocessing, training and the choice of the threshold
    only use the other k-1 folds. The test fold is only used to compute the
    metrics, so the scores are not biased by any tuning.

    Args:
        models: dict {name: (train, score)} where train(y, tx) returns w
            and score(tx, w) returns one score per sample.

    Returns:
        dict {name: list of metric dicts, one per fold}
    """

    k_indices = build_k_indices(len(y), k_fold, seed)
    results = {name: [] for name in models}

    for k in range(k_fold):
        test_idx = k_indices[k]
        train_idx = np.concatenate([k_indices[j] for j in range(k_fold) if j != k])

        # Preprocessing statistics come from the training folds only
        x_tr_clean, x_te_clean = preprocess_data(x[train_idx], x[test_idx])
        tx_tr = np.c_[np.ones(len(x_tr_clean)), x_tr_clean]
        tx_te = np.c_[np.ones(len(x_te_clean)), x_te_clean]
        y_tr, y_te = y[train_idx], y[test_idx]

        for name, (train, score) in models.items():
            w = train(y_tr, tx_tr)

            # Threshold chosen on the training folds, not on the test fold
            threshold, _ = best_threshold(y_tr, score(tx_tr, w))

            scores_te = score(tx_te, w)
            metrics = compute_metrics(y_te, predict_labels(scores_te, threshold))
            metrics["f1_at_05"] = compute_metrics(y_te, predict_labels(scores_te, 0.5))["f1"]
            metrics["threshold"] = threshold
            results[name].append(metrics)

            print(f"  fold {k + 1}/{k_fold}  {name:30s} F1={metrics['f1']:.4f}")

    return results


if __name__ == "__main__":

    # 1-5. Load, convert labels, split, preprocess, add the bias column
    x_train, y_train = load_data()
    y = (y_train == 1).astype(float)

    x_tr, x_val, y_tr, y_val = split_data(x_train, y)
    x_tr_clean, x_val_clean = preprocess_data(x_tr, x_val)

    tx_tr = np.c_[np.ones(len(x_tr_clean)), x_tr_clean]
    tx_val = np.c_[np.ones(len(x_val_clean)), x_val_clean]
    initial_w = np.zeros(tx_tr.shape[1])

    # 6. Logistic regression: try every (gamma, max_iters) pair
    print("Logistic regression")
    results = []
    for gamma in [0.01, 0.1, 0.5, 1.0]:
        for max_iters in [100, 300, 1000]:
            w, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
            threshold, f1 = validation_f1(w, tx_val, y_val)
            results.append((gamma, max_iters, threshold, f1))
            print(f"  gamma={gamma}  max_iters={max_iters}  threshold={threshold:.2f}  F1={f1:.4f}")

    best_gamma, best_iters, _, best_f1 = max(results, key=lambda r: r[3])
    print(f"Best: gamma={best_gamma}  max_iters={best_iters}  F1={best_f1:.4f}")

    # 7. Regularized logistic regression: try several lambdas,
    #    keeping the best gamma and max_iters found above
    print("Regularized logistic regression")
    reg_results = []
    for lambda_ in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, best_iters, best_gamma)
        threshold, f1 = validation_f1(w, tx_val, y_val)
        reg_results.append((lambda_, threshold, f1))
        print(f"  lambda={lambda_}  threshold={threshold:.2f}  F1={f1:.4f}")

    best_lambda, _, best_reg_f1 = max(reg_results, key=lambda r: r[2])
    print(f"Best: lambda={best_lambda}  F1={best_reg_f1:.4f}")

    # 8. Ridge regression (baseline): try a few lambdas
    print("Ridge regression")
    ridge_results = []
    for lambda_ in [1e-5, 1e-4, 1e-3, 1e-2]:
        w, loss = ridge_regression(y_tr, tx_tr, lambda_)
        threshold, f1 = best_threshold(y_val, tx_val @ w)
        ridge_results.append((lambda_, f1))
        print(f"  lambda={lambda_}  threshold={threshold:.2f}  F1={f1:.4f}")

    best_ridge_lambda, _ = max(ridge_results, key=lambda r: r[1])

    # 9. Compare the best version of each model on the same validation set
    print("Model comparison (validation set)")
    print_metrics("Always predict 0", y_val, np.zeros(len(y_val)))

    w, _ = least_squares(y_tr, tx_tr)
    print_metrics("Least squares", y_val, tx_val @ w)

    w, _ = ridge_regression(y_tr, tx_tr, best_ridge_lambda)
    print_metrics(f"Ridge (lambda={best_ridge_lambda})", y_val, tx_val @ w)

    w, _ = logistic_regression(y_tr, tx_tr, initial_w, best_iters, best_gamma)
    print_metrics("Logistic regression", y_val, sigmoid(tx_val @ w))

    w, _ = reg_logistic_regression(y_tr, tx_tr, best_lambda, initial_w, best_iters, best_gamma)
    print_metrics(f"Reg. logistic (lambda={best_lambda})", y_val, sigmoid(tx_val @ w))

        # 10. 5-fold cross-validation of the selected models
    print("5-fold cross-validation")
    models = {
        "Always predict 0": (
            lambda y_, tx: None,
            lambda tx, w: np.zeros(len(tx)),
        ),
        "Least squares": (
            lambda y_, tx: least_squares(y_, tx)[0],
            linear_scores,
        ),
        f"Ridge (lambda={best_ridge_lambda})": (
            lambda y_, tx: ridge_regression(y_, tx, best_ridge_lambda)[0],
            linear_scores,
        ),
        "Logistic regression": (
            lambda y_, tx: logistic_regression(
                y_, tx, np.zeros(tx.shape[1]), best_iters, best_gamma
            )[0],
            logistic_scores,
        ),
        f"Reg. logistic (lambda={best_lambda})": (
            lambda y_, tx: reg_logistic_regression(
                y_, tx, best_lambda, np.zeros(tx.shape[1]), best_iters, best_gamma
            )[0],
            logistic_scores,
        ),
    }

    cv_results = cross_validation(x_train, y, models)

    print("Cross-validation summary (mean +- std over the 5 folds)")
    for name, folds in cv_results.items():
        f1 = [m["f1"] for m in folds]
        print(
            f"  {name:30s} F1={np.mean(f1):.4f} +- {np.std(f1):.4f}  "
            f"F1@0.5={np.mean([m['f1_at_05'] for m in folds]):.4f}  "
            f"precision={np.mean([m['precision'] for m in folds]):.4f}  "
            f"recall={np.mean([m['recall'] for m in folds]):.4f}  "
            f"accuracy={np.mean([m['accuracy'] for m in folds]):.4f}"
        )