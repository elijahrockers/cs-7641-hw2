import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# BEANS GENETIC
pmap = [(10, "black", "pop=10"),
        (20, "yellow", "pop=20"),
        (30, "red", "pop=30")]

mutrate = [0.2, 0.4]

ga = pd.read_csv("beans_GA_NNGSRunner_results.csv")
for mut in mutrate:
    plt.figure()
    plt.title("Beans - Mutation Rate: " + str(mut))
    for pop, color, label in pmap:
        df = ga.loc[ga['param_pop_size'] == pop]
        df = df.loc[ga["param_mutation_rates"] == mut]
        X = df["Iteration"]
        y = df["Fitness"]
        
        plt.plot(X, y, c=color, label=label)

    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("beans_GA_results_" + str(mut) + ".png")


print("Exiting")


