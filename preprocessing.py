import numpy as np

MAX_MISSING = 0.80  # columns with more missing values than this are dropped
MAX_CATEGORIES = 10  # columns with 3 to 10 distinct answers are one-hot encoded


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


def missing_indicators(x_train, x_test):
    """One 0/1 column per missing-value pattern ("was this answer missing?").
    Questions skipped together give identical patterns: kept only once."""

    has_nan = np.any(np.isnan(x_train), axis=0)
    ind_train = np.isnan(x_train[:, has_nan])
    ind_test = np.isnan(x_test[:, has_nan])

    _, first = np.unique(ind_train, axis=1, return_index=True)
    first = np.sort(first)

    return ind_train[:, first].astype(float), ind_test[:, first].astype(float)


def one_hot_encode(x_train, x_test, columns):
    """One 0/1 column per answer of each given column (nan -> all zeros)."""

    hot_train, hot_test = [], []

    for j in columns:
        values = np.unique(x_train[~np.isnan(x_train[:, j]), j])
        hot_train.append((x_train[:, [j]] == values).astype(float))
        hot_test.append((x_test[:, [j]] == values).astype(float))

    return np.hstack(hot_train), np.hstack(hot_test)


def preprocess_data(x_train, x_test):
    """Clean the raw data (Id column included). Every statistic is computed
    on x_train only, then applied to x_test. No bias column is added."""

    # Remove the Id column
    x_train, x_test = x_train[:, 1:], x_test[:, 1:]

    # "Don't know" / "refused" -> nan, "none" -> 0
    x_train, x_test = replace_special_codes(x_train, x_test)

    # Drop the columns with too many missing values
    keep = np.mean(np.isnan(x_train), axis=0) <= MAX_MISSING
    x_train, x_test = x_train[:, keep], x_test[:, keep]

    # Few distinct answers = categorical column
    n_values = np.array([len(np.unique(c[~np.isnan(c)])) for c in x_train.T])
    categorical = (n_values > 2) & (n_values <= MAX_CATEGORIES)

    # Built before imputation, while the nan values are still visible
    ind_train, ind_test = missing_indicators(x_train, x_test)
    hot_train, hot_test = one_hot_encode(x_train, x_test, np.flatnonzero(categorical))

    # Median imputation and standardization
    medians = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), medians, x_train)
    x_test = np.where(np.isnan(x_test), medians, x_test)
    x_train, x_test = standardize(x_train, x_test)

    # The one-hot columns replace the original categorical columns
    x_train = np.hstack([x_train[:, ~categorical], ind_train, hot_train])
    x_test = np.hstack([x_test[:, ~categorical], ind_test, hot_test])

    return standardize(x_train, x_test)