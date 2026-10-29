# CS-433 Project 1: Predicting heart disease (MICHD) from BRFSS data

Binary classification of whether a person has coronary heart disease or a
myocardial infarction (MICHD), from their answers to the Behavioral Risk
Factor Surveillance System (BRFSS) survey. Everything is implemented with
NumPy only.

**Team:** Nada El Ghannouchi, Kenza Hassayoune

## Final model

| Step | Choice |
|---|---|
| Preprocessing | BRFSS special codes ("don't know" / "refused" → missing, "none" → 0), drop columns with more than 99.9% missing values, missing-value indicators, one-hot encoding of categorical columns (3 to 10 answers), median imputation, standardization |
| Model | Logistic regression trained with gradient descent (γ = 0.5, 1000 iterations, w₀ = 0), labels mapped from {-1, 1} to {0, 1} |
| Decision threshold | Chosen to maximize F1 on the training set (0.20 instead of 0.5, since only about 9% of the samples are positive) |
| 5-fold CV F1 | 0.439 ± 0.005 |
| AIcrowd F1 | 0.453, accuracy 0.880 (submission ID 338182) |

## Setup

Only NumPy is needed (pytest for the tests):

```bash
conda create -n ml-project1 python=3.11 numpy pytest
conda activate ml-project1
```

## Data

The dataset is not included in the repository. Download it from the AIcrowd
challenge page and put the files at the root of the repository:

```
CS433-ML-Project1/
├── x_train.csv
├── y_train.csv
├── x_test.csv
├── run.py
└── ...
```

## Reproducing the submission

```bash
python run.py
```

This trains the final model on the whole training set and writes
`submission.csv` (labels in {-1, 1}), the file submitted on AIcrowd. Nothing
in the pipeline is random (zero initialization, full-batch gradient descent),
so every run produces the same file. It takes a few minutes, mostly to read
the CSV files.

## Reproducing the experiments

```bash
python training.py                   # hyperparameter search, model comparison, 5-fold CV
python preprocessing_experiments.py  # preprocessing ablation with 5-fold CV
python improvement_experiments.py    # features with many missing values, class weights
```

The first two scripts use the first version of the preprocessing (features
with more than 80% missing values dropped), as in the tables of the report.

## Files

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression` |
| `preprocessing.py` | `preprocess_data(x_train, x_test)`: the final preprocessing, with all statistics computed on the training data only |
| `evaluation.py` | Accuracy, precision, recall, F1, and the search for the F1-maximizing threshold |
| `training.py` | Train/validation split, hyperparameter grids, model comparison and 5-fold cross-validation |
| `preprocessing_experiments.py` | Effect of each preprocessing step, measured with 5-fold cross-validation |
| `improvement_experiments.py` | Effect of keeping features with many missing values and of class weights, measured with 5-fold cross-validation |
| `run.py` | Trains the final model and creates `submission.csv` |
| `helpers.py` | Loading and submission helpers provided by the course |
| `exploration.py` | First look at the data (shape, missing values) |
| `test_implementations.py`, `test_logistic.py`, `test_preprocessing.py` | Sanity checks of the methods and of the preprocessing |