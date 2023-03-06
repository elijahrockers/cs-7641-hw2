import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulated Annealing
# iterations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

tmap = [(0.1, "black", "temp=0.1"),
        (0.5, "yellow", "temp=0.5"),
        (0.75, "red", "temp=0.75"),
        (1.0, "blue", "temp=1.0"),
        (2.0, "green", "temp=2.0"),
        (5.0, "brown", "temp=5.0")]

fourp = pd.read_csv("four_peaks_SA_curves.csv")
plt.figure()
for temp, color, label in tmap:
    df = fourp.loc[fourp['Temperature'] == temp]
    X = df["Iteration"]
    y = df["Fitness"]
    
    plt.plot(X, y, c=color, label=label)

plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title("Four Peaks")
plt.legend()
plt.savefig("four_peaks_SA_it_v_fit.png")

knap = pd.read_csv("knapsack_SA_curves.csv")
plt.figure()
for temp, color, label in tmap:
    df = knap.loc[knap['Temperature'] == temp]
    X = df["Iteration"]
    y = df["Fitness"]
    
    plt.plot(X, y, c=color, label=label)

plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title("Knapsack")
plt.legend()
plt.savefig("knap_SA_it_v_fit.png")

queens = pd.read_csv("queens_SA_curves.csv")
plt.figure()
for temp, color, label in tmap:
    df = queens.loc[queens['Temperature'] == temp]
    X = df["Iteration"]
    y = df["Fitness"]
    
    plt.plot(X, max(y)-y, c=color, label=label)

plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title("Queens")
plt.legend()
plt.savefig("queens_SA_it_v_fit.png")

print("Exiting")


