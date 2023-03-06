import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulated Annealing
# iterations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

pmap = [(150, "black", "pop=150"),
        (200, "yellow", "pop=200"),
        (300, "red", "pop=300")]

kpct = [0.25, 0.5, 0.75]

fourp = pd.read_csv("four_peaks_MIMIC_curves.csv")
for mut in kpct:
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Four Peaks - Keep Percent: " + str(mut))
    for pop, color, label in pmap:
        df = fourp.loc[fourp['Population Size'] == pop]
        df = df.loc[fourp["Keep Percent"] == mut]
        X = df["Iteration"]
        y = df["Fitness"]
        
        plt.plot(X, y, c=color, label=label)

    plt.legend()
    plt.savefig("four_peaks_MIMIC_it_v_fit_" + str(mut) + ".png")

knap = pd.read_csv("knapsack_MIMIC_curves.csv")
for mut in kpct:
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Knapsack - Keep Percent: " + str(mut))
    for pop, color, label in pmap:
        df = knap.loc[knap['Population Size'] == pop]
        df = df.loc[knap["Keep Percent"] == mut]
        X = df["Iteration"]
        y = df["Fitness"]
        
        plt.plot(X, y, c=color, label=label)

    plt.legend()
    plt.savefig("knapsack_MIMIC_it_v_fit_" + str(mut) + ".png")

queens = pd.read_csv("queens_MIMIC_curves.csv")
for mut in kpct:
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Queens - Keep Percent: " + str(mut))
    for pop, color, label in pmap:
        df = queens.loc[queens['Population Size'] == pop]
        df = df.loc[queens["Keep Percent"] == mut]
        X = df["Iteration"]
        y = df["Fitness"]
        
        plt.plot(X, max(y)-y, c=color, label=label)

    plt.legend()
    plt.savefig("queens_MIMIC_it_v_fit_" + str(mut) + ".png")

print("Exiting")


