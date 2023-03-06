import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import mlrose_hiive as mlrose
from mlrose_hiive import NNGSRunner, relu

SEED=2

# Load dataset
df = pd.read_csv('nn_data/beans.csv')
df = df.sample(frac=0.2, random_state=SEED)
print(len(df))

# Preprocessing
X = df.drop('Class', axis = 1)
attrs = X.columns.tolist()
stdscaler = StandardScaler()
X[attrs] = stdscaler.fit_transform(X[attrs])
le = LabelEncoder()
df['Class'] = le.fit_transform(df.Class.values)
mapping = dict(zip(le.classes_, range(len(le.classes_))))
y = df['Class']

# Split data into training and testing 70:30
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), random_state=SEED, test_size = 0.3)

# One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

# grid_search_parameters = {
#     'learning_rate': [1e-2, 1e-1, 1],                       # nn params
#     'activation': [mlrose.relu, mlrose.tanh],            # nn params
#     'pop_size': [10, 20, 30],
#     'mutation_rates': [0.2, 0.8]
# }
grid_search_parameters = {
    'learning_rate': [1e-1, 1, 10, 100],                       # nn params
    'activation': [mlrose.relu, mlrose.tanh],            # nn params
    'pop_size': [10, 20, 30],
    'mutation_rates': [0.2, 0.4]
}

runner = NNGSRunner(
    x_train=X_train,
    y_train=y_train_hot,
    x_test=X_test,
    y_test=y_test_hot,
    experiment_name='nn_test_ga',
    algorithm=mlrose.algorithms.ga.genetic_alg,
    grid_search_parameters=grid_search_parameters,
    iteration_list=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    hidden_layer_sizes=[[500, 500, 10]],
    bias=True,
    early_stopping=True,
    clip_max=5,
    max_attempts=100,
    n_jobs=2,
    cv=2,
    seed=SEED,
    output_directory=None
)

start = time.time()
run_stats_df, curves_df, cv_results_df, grid_search_cv = runner.run()
end = time.time()
print("Elapsed: ", (end - start))

run_stats_df.to_csv(os.path.join("experiments", "beans_GA_NNGSRunner_stats.csv"), header=True)
curves_df.to_csv(os.path.join("experiments", "beans_GA_NNGSRunner_curves.csv"), header=True)
cv_results_df.to_csv(os.path.join("experiments", "beans_GA_NNGSRunner_results.csv"), header=True)
with open('experiments/beans_GA_NNGSRunner_grid.txt', "w") as f:
    print("N: " + str(len(df)), file=f)
    print("Total elapsed: " + str(end - start), file=f) 
    print(grid_search_cv, file=f)

print("Exiting.")
