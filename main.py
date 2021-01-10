import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# Load data
data = pd.read_csv("./train.csv")

# Preprocess
trainable = ["Pclass", "Age", "Fare"]
passengers = []
survived = []

for i in range(len(data)):
	passenger = []
	for t in trainable:
		passenger.append(data[t][i])
	passengers.append(passenger)
	survived.append(data["Survived"][i])

print(survived)