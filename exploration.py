import numpy as np

x_train = np.genfromtxt("x_train.csv", delimiter=",", skip_header=1)

print("Dimensions du dataset :", x_train.shape)

print("Premières lignes :")
print(x_train[:5])


missing_values = np.isnan(x_train).sum()

print("Valeurs manquantes :", missing_values)

total_values = x_train.size

missing_percentage = (missing_values / total_values) * 100

print("Pourcentage manquant :", missing_percentage)

missing_per_column = np.isnan(x_train).sum(axis=0)

print("Colonnes avec valeurs manquantes :", np.sum(missing_per_column > 0))


with open("x_train.csv", "r") as f:
    column_names = f.readline().strip().split(",")

missing_per_column = np.isnan(x_train).mean(axis=0) * 100

sorted_indices = np.argsort(missing_per_column)[::-1]

print("\nTop 15 colonnes avec le plus de NaN :")

for i in sorted_indices[:15]:
    print(column_names[i], ":", round(missing_per_column[i], 2), "%")

high_missing = np.sum(missing_per_column > 80)

print("\nColonnes avec plus de 80 % de NaN :", high_missing)
