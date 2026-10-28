"""Measure the effect of preprocessing improvements with 5-fold cross-validation.

Each variant adds changes to the basic preprocessing (median imputation +
standardization), then trains logistic regression on the same folds as
training.py, so the F1 scores can be compared fold by fold with the baseline.
The best variant ("Codes + one-hot + indicators") is the final preprocessing,
implemented in preprocessing.preprocess_data.
"""

import numpy as np

from evaluation import best_threshold, compute_metrics, predict_labels
from implementations import logistic_regression, sigmoid
from training import build_k_indices, load_data

GAMMA = 0.5
MAX_ITERS = 1000
K_FOLD = 5
MAX_CATEGORIES = 10


def standardize(x_train, x_test):
    """Mean 0 and std 1, using the training statistics (std 0 is set to 1)."""

    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)
    stds[stds == 0] = 1

    return (x_train - means) / stds, (x_test - means) / stds


def replace_special_codes(x_train, x_test):
    """Replace the BRFSS answer codes. The largest value of a column (in the
    training set) tells how many digits its answers have:
      max 9:    9 (refused) -> nan, and 7 (don't know) -> nan unless 8 is
                also an answer (then 7 is probably a real category)
      max 99:   77, 99 -> nan, 88 (none) -> 0
      max 999:  777, 999 -> nan, 888 (none) -> 0
      max 9999: 7777, 9999 -> nan
    """

    x_train, x_test = x_train.copy(), x_test.copy()

    for j in range(x_train.shape[1]):
        column = x_train[:, j]
        if np.all(np.isnan(column)):
            continue

        largest = np.nanmax(column)
        if largest == 9:
            missing = [9] if np.any(column == 8) else [7, 9]
            none = None
        elif largest == 99:
            missing, none = [77, 99], 88
        elif largest == 999:
            missing, none = [777, 999], 888
        elif largest == 9999:
            missing, none = [7777, 9999], None
        else:
            continue

        for x in (x_train, x_test):
            x[np.isin(x[:, j], missing), j] = np.nan
            if none is not None:
                x[x[:, j] == none, j] = 0

    return x_train, x_test


def preprocess_v2(
    x_train,
    x_test,
    special_codes=False,
    missing_indicators=False,
    one_hot=False,
    squares=False,
):
    """Basic preprocessing plus optional improvements, switched on by flags.
    With special_codes, missing_indicators and one_hot set (and squares not),
    it gives the same result as preprocessing.preprocess_data."""

    # Remove the Id column
    x_train, x_test = x_train[:, 1:], x_test[:, 1:]

    if special_codes:
        x_train, x_test = replace_special_codes(x_train, x_test)

    # Drop the columns with more than 80% missing values
    keep = np.mean(np.isnan(x_train), axis=0) <= 0.80
    x_train, x_test = x_train[:, keep], x_test[:, keep]

    # Number of distinct answers per column: few answers = categorical
    n_values = np.array([len(np.unique(c[~np.isnan(c)])) for c in x_train.T])
    categorical = (n_values > 2) & (n_values <= MAX_CATEGORIES)
    continuous = n_values > MAX_CATEGORIES

    extra_train, extra_test = [], []

    if missing_indicators:
        # One 0/1 column per missing-value pattern ("was this answer missing?").
        # Questions skipped together give identical patterns: kept only once.
        has_nan = np.any(np.isnan(x_train), axis=0)
        ind_train = np.isnan(x_train[:, has_nan])
        ind_test = np.isnan(x_test[:, has_nan])
        _, first = np.unique(ind_train, axis=1, return_index=True)
        first = np.sort(first)
        extra_train.append(ind_train[:, first].astype(float))
        extra_test.append(ind_test[:, first].astype(float))

    if one_hot:
        # One 0/1 column per answer of each categorical column
        for j in np.flatnonzero(categorical):
            values = np.unique(x_train[~np.isnan(x_train[:, j]), j])
            extra_train.append((x_train[:, [j]] == values).astype(float))
            extra_test.append((x_test[:, [j]] == values).astype(float))

    # Median imputation and standardization, as in preprocess_data
    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_test = np.where(np.isnan(x_test), medians, x_test)
    x_train, x_test = standardize(x_train, x_test)

    if squares:
        # Squares of the standardized continuous columns (non-linear effects)
        extra_train.append(x_train[:, continuous] ** 2)
        extra_test.append(x_test[:, continuous] ** 2)

    if one_hot:
        # The one-hot columns replace the original categorical columns
        x_train, x_test = x_train[:, ~categorical], x_test[:, ~categorical]

    x_train = np.hstack([x_train] + extra_train)
    x_test = np.hstack([x_test] + extra_test)

    return standardize(x_train, x_test)


if __name__ == "__main__":

    x_train, y_train = load_data()
    y = (y_train == 1).astype(float)
    k_indices = build_k_indices(len(y), K_FOLD)  # same folds as training.py

    # All the variants of the preprocessing ablation table of the report
    variants = {
        "Baseline (median + standardize)": {},
        "+ special codes": {"special_codes": True},
        "+ missing indicators": {"missing_indicators": True},
        "+ one-hot categorical": {"one_hot": True},
        "+ squared features": {"squares": True},
        "Codes + one-hot": {"special_codes": True, "one_hot": True},
        "Codes + one-hot + indicators": {
            "special_codes": True,
            "one_hot": True,
            "missing_indicators": True,
        },
        "All four": {
            "special_codes": True,
            "missing_indicators": True,
            "one_hot": True,
            "squares": True,
        },
    }

    baseline_f1 = None
    for name, options in variants.items():
        f1_scores, losses = [], []

        for k in range(K_FOLD):
            test_idx = k_indices[k]
            train_idx = np.concatenate([k_indices[j] for j in range(K_FOLD) if j != k])

            x_tr_clean, x_te_clean = preprocess_v2(
                x_train[train_idx], x_train[test_idx], **options
            )
            tx_tr = np.c_[np.ones(len(x_tr_clean)), x_tr_clean]
            tx_te = np.c_[np.ones(len(x_te_clean)), x_te_clean]
            y_tr, y_te = y[train_idx], y[test_idx]

            w, loss = logistic_regression(
                y_tr, tx_tr, np.zeros(tx_tr.shape[1]), MAX_ITERS, GAMMA
            )

            # Threshold chosen on the training folds, F1 measured on the test fold
            threshold, _ = best_threshold(y_tr, sigmoid(tx_tr @ w))
            y_pred = predict_labels(sigmoid(tx_te @ w), threshold)

            f1_scores.append(compute_metrics(y_te, y_pred)["f1"])
            losses.append(loss)
            print(
                f"  {name}: fold {k + 1}/{K_FOLD}  F1={f1_scores[-1]:.4f}", flush=True
            )

        f1_scores = np.array(f1_scores)
        line = (
            f"{name:32s} features={tx_tr.shape[1]:4d}  train loss={np.mean(losses):.4f}  "
            f"F1={f1_scores.mean():.4f} +- {f1_scores.std():.4f}"
        )
        if baseline_f1 is None:
            baseline_f1 = f1_scores
        else:
            gain = f1_scores - baseline_f1
            line += f"  gain={gain.mean():+.4f} (better in {np.sum(gain > 0)}/{K_FOLD} folds)"
        print(line, flush=True)
