import numpy as np
import pandas as pd

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input as KInput, Dense as KDense

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
		passenger.append(np.float(data[t][i]))
	passengers.append(passenger)
	survived.append(np.float(data["Survived"][i]))

passengers = np.array(passengers)
survived = np.array(survived)

# Model
model_input = KInput(3)
model_layer_0 = KDense(64, activation="relu")(model_input)
model_output = KDense(1, activation="sigmoid")(model_layer_0)

model = Model(inputs=model_input, outputs=model_output)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x=passengers, y=survived, batch_size=10, epochs=5)
model.summary()
 
print(model.predict([[2, 64, 154]]))