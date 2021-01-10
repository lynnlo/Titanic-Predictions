
# Imports
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input as KInput, Dense as KDense

from matplotlib import pyplot as plt

# Load data
data = pd.read_csv("./train.csv")

# Preprocess
trainable = ["Pclass", "Age", "Fare"]
passengers = []
survived = []
scalers = []

for i in trainable:
	scaler = MinMaxScaler(feature_range=(0,1))
	scaler.fit(np.array(data[i]).reshape(-1, 1))
	scalers.append(scaler)

for i in range(len(data)):
	passenger = []
	# Scale per value
	for ti,t in enumerate(trainable):
		value = np.array(np.float(data[t][i])).reshape(1, -1) if not np.isnan(data[t][i]) else [[0]]
		passenger.append(scalers[ti].transform(value)[0][0])
	passengers.append(passenger)
	survived.append(np.float(data["Survived"][i]))

passengers = np.array(passengers) # Values
survived = np.array(survived) # Labels

print(passengers)

# Model
model_input = KInput(3) # Input
model_layer_0 = KDense(64, activation="relu")(model_input) # Hidden
model_output = KDense(1, activation="sigmoid")(model_layer_0) # Output

model = Model(inputs=model_input, outputs=model_output)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x=passengers, y=survived, batch_size=10, epochs=20)
model.summary()

print(model.predict(passengers[0].reshape(1, 3)))
print(survived[0])