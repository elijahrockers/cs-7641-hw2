import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulated Annealing
# iterations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

pmap = [(150, "black", "pop=150"),
        (200, "yellow", "pop=200"),
        (300, "red", "pop=300")]

mutrate = [0.4, 0.5, 0.6]

fourp = pd.read_csv("four_peaks_GA_curves.csv")
for mut in [0.4, 0.5, 0.6]:
    plt.figure()
    plt.title("Four Peaks - Mutation Rate: " + str(mut))
    for pop, color, label in pmap:
        df = fourp.loc[fourp['Population Size'] == pop]
        df = df.loc[fourp["Mutation Rate"] == mut]
        X = df["Iteration"]
        y = df["Fitness"]
        
        plt.plot(X, y, c=color, label=label)

    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("four_peaks_GA_it_v_fit_" + str(mut) + ".png")

knap = pd.read_csv("knapsack_GA_curves.csv")
for mut in [0.4, 0.5, 0.6]:
    plt.figure()
    plt.title("Knapsack - Mutation Rate: " + str(mut))
    for pop, color, label in pmap:
        df = knap.loc[knap['Population Size'] == pop]
        df = df.loc[knap["Mutation Rate"] == mut]
        X = df["Iteration"]
        y = df["Fitness"]
        
        plt.plot(X, y, c=color, label=label)

    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("knapsack_GA_it_v_fit_" + str(mut) + ".png")

queens = pd.read_csv("queens_GA_curves.csv")
for mut in [0.4, 0.5, 0.6]:
    plt.figure()
    plt.title("Queens - Mutation Rate: " + str(mut))
    for pop, color, label in pmap:
        df = queens.loc[queens['Population Size'] == pop]
        df = df.loc[queens["Mutation Rate"] == mut]
        X = df["Iteration"]
        y = df["Fitness"]
        
        plt.plot(X, max(y)-y, c=color, label=label)

    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("queens_GA_it_v_fit_" + str(mut) + ".png")


print("Exiting")


