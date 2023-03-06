import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RHC
fourp = pd.read_csv("four_peaks_RHC_curves.csv")
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Fitness")
df = fourp
X = df["Iteration"]
y = df["Fitness"]
    
plt.plot(X, y)

plt.title("Four Peaks")
plt.savefig("four_peaks_RHC_it_v_fit.png")

knap = pd.read_csv("knapsack_RHC_curves.csv")
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Fitness")
df = knap
X = df["Iteration"]
y = df["Fitness"]
    
plt.plot(X, y)

plt.title("Knapsack")
plt.savefig("knapsack_RHC_it_v_fit.png")

queens = pd.read_csv("queens_RHC_curves.csv")
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Fitness")
df = queens
X = df["Iteration"]
y = df["Fitness"]
    
plt.plot(X, max(y)-y)

plt.title("Queens")
plt.savefig("queens_RHC_it_v_fit.png")
print("Exiting")
