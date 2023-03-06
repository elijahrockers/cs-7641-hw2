import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import time
import os

from mlrose_hiive import SARunner, GARunner 
from mlrose_hiive import NNGSRunner, RHCRunner, MIMICRunner

SEED = 2
EXPERIMENT = "tsp_"

# Traveling Salesman Problem: spiral within a 6x6 grid, best (lowest) fitness score is ~34.6
coords_list = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
               (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
               (5, 6), (4, 6), (3, 6), (2, 6),  (1, 6), (0, 6),
               (0, 5), (0, 4), (0, 3), (0, 2),
               (1, 2), (2, 2), (3, 2), (4, 2),
               (4, 3), (4, 4), (3, 4), (2, 4), (2, 3)]
fitness = mlrose.TravellingSales(coords=coords_list)
problem = mlrose.TSPOpt(length=len(coords_list), fitness_fn=fitness, maximize=False)
problem.set_mimic_fast_mode(True)

##### SA
print("Simulated Annealing...")
runner = SARunner(problem=problem,
              experiment_name=EXPERIMENT + "SA",
              output_directory=None,
              seed=SEED,
              iteration_list=2 ** np.arange(14),
              max_attempts=500,
              temperature_list=[0.1, 0.5, 0.75, 1.0, 2.0, 5.0],
              decay_list=[mlrose.GeomDecay])

# the two data frames will contain the results
df_run_stats, df_run_curves = runner.run()
df_run_stats.to_csv(os.path.join("experiments", "tsp_SA_stats.csv"), header=True)
df_run_curves.to_csv(os.path.join("experiments", "tsp_SA_curves.csv"), header=True)

##### RHC
print("Randomized Hill Climbing...")
runner = RHCRunner(problem=problem,
              experiment_name=EXPERIMENT + "RHC",
              output_directory=None,
              seed=SEED,
              iteration_list=2 ** np.arange(14),
              max_attempts=500,
              restart_list=[0])

# the two data frames will contain the results
df_run_stats, df_run_curves = runner.run()
df_run_stats.to_csv(os.path.join("experiments", "tsp_RHC_stats.csv"), header=True)
df_run_curves.to_csv(os.path.join("experiments", "tsp_RHC_curves.csv"), header=True)

##### GA
print("Genetic Algorithms...")
runner = GARunner(problem=problem,
              experiment_name=EXPERIMENT + "GA",
              output_directory=None,
              seed=SEED,
              iteration_list=2 ** np.arange(14),
              max_attempts=500,
              population_sizes=[150, 200, 300],
              mutation_rates=[0.4, 0.5, 0.6])

# the two data frames will contain the results
df_run_stats, df_run_curves = runner.run()
df_run_stats.to_csv(os.path.join("experiments", "tsp_GA_stats.csv"), header=True)
df_run_curves.to_csv(os.path.join("experiments", "tsp_GA_curves.csv"), header=True)

##### MIMIC
print("MIMIC Algorithms...")
runner = MIMICRunner(problem=problem,
              experiment_name=EXPERIMENT + "MIMIC",
              output_directory=None,
              seed=SEED,
              iteration_list=2 ** np.arange(14),
              max_attempts=500,
              population_sizes=[150, 200, 300],
              keep_percent_list=[0.25, 0.50, 0.75],
              use_fast_mimic=True)

# the two data frames will contain the results
df_run_stats, df_run_curves = runner.run()
df_run_stats.to_csv(os.path.join("experiments", "tsp_MIMIC_stats.csv"), header=True)
df_run_curves.to_csv(os.path.join("experiments", "tsp_MIMIC_curves.csv"), header=True)
print("")
