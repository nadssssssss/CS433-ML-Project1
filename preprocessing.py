import numpy as np


def preprocess_data(x_train, x_test):
    x_train = x_train[:, 1:].copy()
    x_test = x_test[:, 1:].copy()

    missing_ratio = np.mean(np.isnan(x_train), axis=0)

    keep_columns = missing_ratio <= 0.80

    x_train = x_train[:, keep_columns]
    x_test = x_test[:, keep_columns]

    medians = np.nanmedian(x_train, axis=0)

    x_train = np.where(np.isnan(x_train), medians, x_train)

    x_test = np.where(np.isnan(x_test), medians, x_test)

    means = np.mean(x_train, axis=0)
    stds = np.std(x_train, axis=0)

    stds[stds == 0] = 1

    x_train = (x_train - means) / stds
    x_test = (x_test - means) / stds

    return x_train, x_test
