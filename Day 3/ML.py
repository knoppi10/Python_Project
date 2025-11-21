import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score

# Set a seed so the results are somewhat replicable
np.random.seed(42)
# Set this to the folder containing the datasets
# IMPORTANT: You need to use forward slashes ("/") not backward slashes ("\")
# The current folder.
path = "./"

"""# Random Forests"""

df = pd.read_csv("credit-card-default.csv")
df.info()

print(df['default'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
        df.drop('default', axis=1),
        df['default'],
        test_size = 0.20,
        stratify = df['default'])

from sklearn.ensemble import RandomForestClassifier
# n_estimators: number of trees
forest = RandomForestClassifier(n_estimators = 5)
forest.fit(X_train, y_train)
forest.score(X_test, y_test)

importances = forest.feature_importances_
indices = np.argsort(-importances)
"""
The function np.argsort() in NumPy returns the indices that would sort an array.
It is often used to get the order of elements in an array 
without actually sorting the array itself. 
The indices can then be used to rearrange or rank elements according to their sorted order.
"""

# Create a dataframe just for pretty printing
df_imp = pd.DataFrame(dict(feature=X_train.columns[indices],
                           importance = importances[indices]))
df_imp.head()

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(importances)), df_imp['importance'], color="b", align="center")
plt.xticks(range(len(importances)), df_imp['feature'], rotation=90)
plt.tight_layout()
plt.show()

"""# Confusing Marix and Scoring Metrics"""

forest.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
y_test_pred = forest.predict(X_test)
conf_mat = confusion_matrix(y_test, y_test_pred)
conf_mat

# (TP + TN) / (TP + TN + FP + FN)
print("Accuracy:", (conf_mat[0,0] + conf_mat[1,1]) / np.sum(conf_mat))
# TP / (TP + FP)
print("Precision:", conf_mat[1,1] / np.sum(conf_mat[:,1]))
# TP / (TP + FN)
print("Recall:", conf_mat[1,1] / np.sum(conf_mat[1,:]))
precision = conf_mat[1,1] / np.sum(conf_mat[:,1])
recall = conf_mat[1,1] / np.sum(conf_mat[1,:])
print("F1-score:", (2*precision*recall)/(precision+recall))

from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1 score:", f1_score(y_test, y_test_pred))


import matplotlib.pyplot as plt
import seaborn as sns

# Compute metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Add metrics to the plot
plt.text(-0.4, -0.3, f'Accuracy: {accuracy:.2f}', fontsize=12, color='black', ha='left')
plt.text(-0.4, -0.6, f'Precision: {precision:.2f}', fontsize=12, color='black', ha='left')
plt.text(-0.4, -0.9, f'Recall: {recall:.2f}', fontsize=12, color='black', ha='left')
plt.text(-0.4, -1.2, f'F1 Score: {f1:.2f}', fontsize=12, color='black', ha='left')

plt.tight_layout()
plt.show()



"""# Multi-class Example: Children Nursery Dataset"""

np.random.seed(42)
def my_describe_dataframe(df):
    for col in df.columns:
        if df.dtypes[col] == object:
            # Convert to list to print in one-line
            print(col + " : " + str(df[col].unique().tolist()))
        else:
            # Convert to dictionary to print in one-line but keep names
            print(df[col].describe().to_dict())

nursery = pd.read_csv("nursery-data.csv")
nursery.info()

my_describe_dataframe(nursery)

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
nursery_enc = nursery.copy()
for col in nursery.select_dtypes(include=object).columns:
    label_encoders[col] = LabelEncoder()
    nursery_enc[col] = label_encoders[col].fit_transform(nursery[col])

# Train/test split
X_train, X_test, y_train, y_test = \
    train_test_split(nursery_enc.drop('Application_Status', axis=1),
                     nursery_enc['Application_Status'],
                     test_size = 0.20,
                     stratify = nursery_enc['Application_Status'])

from sklearn.ensemble import RandomForestClassifier
# n_estimators: number of trees
forest = RandomForestClassifier(n_estimators = 5)
forest.fit(X_train, y_train)
forest.score(X_test, y_test)

importances = forest.feature_importances_
indices = np.argsort(-importances)
# Create a dataframe just for pretty printing
df_imp = pd.DataFrame(dict(feature=X_train.columns[indices],
                           importance = importances[indices]))
df_imp.head()

# Another visualization
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(importances)), df_imp['importance'], color="b", align="center")
plt.xticks(range(len(importances)), df_imp['feature'], rotation=90)
plt.tight_layout()
plt.show()

# Sort features by importance
df_imp = df_imp.sort_values(by='importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(
    x='feature',
    y='importance',
    data=df_imp,
    palette="Blues_d"
)

# Add a dashed line to separate the top 5 features
plt.axvline(x=4.5, color='red', linestyle='--', linewidth=1.5, label='Top 5 Features')
plt.title("Feature Importances", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)
plt.xticks(rotation=90, fontsize=10)
plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix
y_test_pred = forest.predict(X_test)
conf_mat = confusion_matrix(y_test, y_test_pred, labels = forest.classes_)
print(conf_mat)

from sklearn.metrics import ConfusionMatrixDisplay
plt.style.use('default')
plt.figure(figsize=(10,10))
disp=ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=label_encoders['Application_Status'].classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(nursery['Application_Status'].value_counts())

from sklearn.metrics import classification_report
print(
  classification_report(
      label_encoders['Application_Status'].inverse_transform(y_test),
      label_encoders['Application_Status'].inverse_transform(y_test_pred)
      )
  )

"""## Quiz"""

from sklearn.metrics import accuracy_score, precision_score,\
                            recall_score, f1_score
                            
y_true = [0,1,0,1,1,0,1,1,1,1]
y_pred = [0,1,1,0,1,1,0,0,1,1]
confusion_matrix(y_true, y_pred)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 score:", f1_score(y_true, y_pred))

"""## Cross-validation with different scores: Default in Credit Card Payments"""

df = pd.read_csv("credit-card-default.zip")
df.info()

print(df['default'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
        df.drop('default', axis=1),
        df['default'],
        test_size = 0.20,
        stratify = df['default'])

from sklearn.ensemble import RandomForestClassifier
# n_estimators: number of trees
forest = RandomForestClassifier(n_estimators = 5)
# K-fold cross-validation
scores = cross_val_score(forest, X_train, y_train, cv=10)
print(f"{len(scores)}-fold CV Score: {scores.mean():.2f} (+/- {scores.std() * 2:2f})")

from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score)

# K-fold cross-validation
scores = cross_val_score(forest, X_train, y_train, cv=10, scoring = f1_scorer)
print(f"{len(scores)}-fold CV Score: {scores.mean():.2f} (+/- {scores.std() * 2:2f})")

"""# Hyper-parameter optimization

## Grid Search
"""

# California House Prices dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
print(housing.DESCR)

from sklearn.model_selection import train_test_split
# 1. Split train/test
X_train, X_test, y_train, y_test = \
    train_test_split(housing.data, housing.target, test_size = 0.10)
# Standardization (z-score normalization)
from sklearn.preprocessing import StandardScaler
# We fit the scaler on the train data and apply to both train and test
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=10, solver='sgd', max_iter=1000)
# K-fold cross-validation
scores = cross_val_score(mlp, X_train_scaled, y_train, cv=10)
print(f"{len(scores)}-fold CV Score: {scores.mean():.2f} (+/- {2*scores.std():2f})")

# Refit using all training data
mlp.fit(X_train_scaled, y_train)
mlp.score(X_test_scaled, y_test)

# MLPRegressor may generate warnings if using too few iterations.
# The solution is to run with more generations (or try alternative ways to preprocess the data).
# However, we silence the warnings here so that the examples do not take too long.
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

mlp = MLPRegressor(max_iter=200)

# Parameters and values to tune
param_grid = dict(hidden_layer_sizes= [10, 25, (5,5), (10,10)],
                  solver = ['lbfgs','sgd'])

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(mlp, param_grid, cv=10)
# Run the search
gs.fit(X_train_scaled, y_train)
print("Best parameters found:", gs.best_params_)
print("Mean CV score of best parameters:", gs.best_score_)
# Before calculating the score, the model is refit using all training data.
print("Score of best parameters on test data:",
      gs.score(X_test_scaled, y_test))

means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
params = gs.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print(f"{mean:.3f} (+/-{2*std:.3f}) for {param}")

"""## Random search"""

np.random.seed(42)
from timeit import default_timer as timer

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: ", i)
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: ", results['params'][candidate], "\n")

import scipy.stats as sp
# sp.randint generates a random number generator
param_dist = dict(hidden_layer_sizes = sp.randint(5,10,(1,1)),
                  solver = ['lbfgs','sgd'])

from sklearn.model_selection import RandomizedSearchCV
n_iter_search = 10
random_search = RandomizedSearchCV(
        MLPRegressor(max_iter=200), param_distributions=param_dist,
        n_iter=n_iter_search, cv=10)

start = timer()
random_search.fit(X_train_scaled, y_train)
print(f"RandomizedSearch took {timer() - start:.2f} seconds for {n_iter_search} hyper-parameter configurations.")
report(random_search.cv_results_)

print("Best parameters set found:", random_search.best_params_)
print("Mean CV score of best parameters:", random_search.best_score_)
# Before calculating the score, the model is refit using all training data.
print("Score of best parameters on test data:",
      random_search.score(X_test_scaled, y_test))

"""## Example of hyper-parameter optimization with other metrics"""

df = pd.read_csv("credit-card-default.zip")
df.info()

print(df['default'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
        df.drop('default', axis=1),
        df['default'],
        test_size = 0.20,
        stratify = df['default'])

from sklearn.ensemble import RandomForestClassifier
# n_estimators: number of trees
forest = RandomForestClassifier(n_estimators = 5)
# K-fold cross-validation
scores = cross_val_score(forest, X_train, y_train, cv=10)
print(f"{len(scores)}-fold CV Score: {scores.mean():.2f} (+/- {2*scores.std():2f})")

from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score)
# K-fold cross-validation
scores = cross_val_score(forest, X_train, y_train, cv=10, scoring = f1_scorer)
print(f"{len(scores)}-fold CV Score: {scores.mean():.2f} (+/- {2*scores.std():2f})")

import scipy.stats as sp
# sp.randint generates a discrete random number generator (RNG)
# sp.uniform generates a continuous uniform RNG
param_dist = dict(n_estimators = sp.randint(5,20, 1),
                  # None is the default value and means maximum depth
                  max_depth = [10, 50, 100, None])              

from sklearn.model_selection import RandomizedSearchCV

n_iter_search = 10
randsearch = RandomizedSearchCV(RandomForestClassifier(),
                                param_distributions = param_dist,
                                n_iter = n_iter_search, cv=10)

randsearch.fit(X_train, y_train)
print("Best parameters set found:", randsearch.best_params_)
print("Mean score of best parameters:", randsearch.best_score_)

from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score)
randsearch = RandomizedSearchCV(RandomForestClassifier(),
                                param_distributions = param_dist,
                                n_iter = n_iter_search, cv=10,
                                scoring = f1_scorer)
randsearch.fit(X_train, y_train)
print("Best parameters set found:", randsearch.best_params_)
print("Mean F1-score of best parameters:", randsearch.best_score_)


"""# Pipelines
The above examples are not completely correct
because we fitted the `StandardScaler` using
the whole training data, however, when doing cross-validation only part of
the training data is used as training data, the rest is used for validation.
"""

np.random.seed(42)
# California House Prices dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
print(housing.DESCR)

from sklearn.model_selection import train_test_split
# 1. Split train/test
X_train, X_test, y_train, y_test = \
    train_test_split(housing.data, housing.target, test_size = 0.10)
# Standardization (z-score normalization)
from sklearn.preprocessing import StandardScaler
# We fit the scaler on the train data and apply to both train and test
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor

import scipy.stats as sp
# sp.randint generates a random number generator
param_dist = dict(hidden_layer_sizes = sp.randint(5,10,(1,1)),
                  solver = ['lbfgs','sgd'])

from sklearn.model_selection import RandomizedSearchCV
n_iter_search = 10
random_search = RandomizedSearchCV(
        MLPRegressor(max_iter=200), param_distributions=param_dist,
        n_iter=n_iter_search, cv=10, random_state=42)
random_search.fit(X_train_scaled, y_train)
print("Best parameters set found:", random_search.best_params_)
print("Mean CV score of best parameters:", random_search.best_score_)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('stdscale', StandardScaler()),
                     ('mlp',  MLPRegressor())])
print(pipeline.named_steps)

print(pipeline.named_steps['stdscale'])

from sklearn import set_config
set_config(display="diagram")
pipeline

# Prefix hyper-parameters with mlp__
pipe_param_dist = dict(mlp__hidden_layer_sizes = sp.randint(5,10,(1,1)),
                       mlp__solver = ['lbfgs','sgd'])

random_search = RandomizedSearchCV(
        pipeline, param_distributions=pipe_param_dist,
        n_iter=n_iter_search, cv=10, random_state=42, n_jobs=2)
random_search.fit(X_train, y_train)
print("Best parameters set found:", random_search.best_params_)
print("Mean CV score of best parameters:", random_search.best_score_)

