# import packages and load data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# import logistic regression libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay

# import random forest libraries
from sklearn.ensemble import RandomForestClassifier

# import cross validation libraries
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv(r"C:\Users\woowe\Downloads\DBA3803\hospital_readmissions.csv")
print(df.head())

# one-hotting independent variables
df2 = pd.get_dummies(df, columns = ['diabetes_med'], drop_first = True, dtype = int)
df3 = pd.get_dummies(df2, columns = ['age'], drop_first = True, dtype = int)
df4 = pd.get_dummies(df3, columns = ['medical_specialty'], drop_first = True, dtype = int)
df5 = pd.get_dummies(df4, columns = ['diag_1'], drop_first = True, dtype = int)
df6 = pd.get_dummies(df5, columns = ['diag_2'], drop_first = True, dtype = int)
df7 = pd.get_dummies(df6, columns = ['diag_3'], drop_first = True, dtype = int)
df8 = pd.get_dummies(df7, columns = ['glucose_test'], drop_first = True, dtype = int)
df9 = pd.get_dummies(df8, columns = ['A1Ctest'], drop_first = True, dtype = int)

# one-hotting change variable
df10 = pd.get_dummies(df9, columns = ['change'], drop_first = True, dtype = int)
print(df10)

# one-hot readmission variable
df11 = pd.get_dummies(df10, columns = ['readmitted'], drop_first = True, dtype = int)
print(df11.head())

# split columns into dependent variable and independent variables
# define features (X) and target (y)
X = df11.drop(['readmitted_yes'], axis=1)
y = df11['readmitted_yes']

# split data points (rows) into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# initialize models
randomforest = RandomForestClassifier(max_depth = 6, random_state = 3)

# define hyperparameters to tune
params = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required to be at a leaf node
}

# set up GridSearchCV
grid_search = GridSearchCV(estimator=randomforest, param_grid=params, cv=5, scoring='roc_auc', verbose=4,
                          return_train_score=True)

# fit the model on the training data
grid_search.fit(X_train, y_train)

# check the best parameters and the corresponding score
best_parameters = grid_search.best_params_
best_auc_score = grid_search.best_score_
print(f"Best Parameters: {best_parameters}")
print(f"Best AUC Score: {best_auc_score}")

# evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_auc = roc_auc_score(y_test, y_pred)
print(f"Test AUC: {test_auc}")