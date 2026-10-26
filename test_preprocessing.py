import numpy as np
from preprocessing import preprocess_data

# Charger les données
x_train = np.genfromtxt("x_train.csv", delimiter=",", skip_header=1)

x_test = np.genfromtxt("x_test.csv", delimiter=",", skip_header=1)

# Appliquer le preprocessing
x_train_clean, x_test_clean = preprocess_data(x_train, x_test)

print("Avant :", x_train.shape)
print("Après :", x_train_clean.shape)

print("NaN restants dans train :", np.isnan(x_train_clean).sum())

print("NaN restants dans test :", np.isnan(x_test_clean).sum())


print("Moyennes des 5 premières colonnes :")
print(np.mean(x_train_clean, axis=0)[:5])

print("Écarts-types des 5 premières colonnes :")
print(np.std(x_train_clean, axis=0)[:5])
