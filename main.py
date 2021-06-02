# Imports
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input as KInput, Dense as KDense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Load data
data = pd.read_csv("./train.csv")

# Preprocess
trainable = ["Pclass", "Age", "Parch", "SibSp", "Fare"]
passengers = []
survived = []
scalers = []

for i in trainable:
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit(np.array(data[i]).reshape(-1, 1))
	scalers.append(scaler)

for i in range(len(data)):
	passenger = []
	# Scale per value
	for ti,t in enumerate(trainable):
		value = np.array(np.float(data[t][i])).reshape(1, -1) if not np.isnan(data[t][i]) else [[0]]
		passenger.append(scalers[ti].transform(value)[0][0])
	passenger.append(1 if data["Sex"][i] == "male" else 0) # Gender
	passengers.append(passenger)
	survived.append(np.float(data["Survived"][i]))

passengers = np.array(passengers) # Values
survived = np.array(survived) # Labels

# Model
new = False
train = True 

if new:
	model_input = KInput(len(trainable)+1) # Input
	model_layer_0 = KDense(128, activation="relu")(model_input) # Hidden
	model_layer_1 = KDense(32, activation="relu")(model_layer_0) # Hidden
	model_output = KDense(1, activation="sigmoid")(model_layer_1) # Output

	model = Model(inputs=model_input, outputs=model_output)
	model.compile(optimizer="adam", loss="binary_crossentropy")

	model.fit(x=passengers, y=survived, batch_size=1, epochs=500) # Training
	model.summary()

	model.save("model")
else:
	model = load_model("model")

	if train:
		model.fit(x=passengers, y=survived, batch_size=1, epochs=250) # Training
		model.summary()

		model.save("model")

def predict(data : np.array):
	return np.round(model.predict(data.reshape(1, len(trainable)+1)))[0][0]

# Prediction
testing_data = pd.read_csv("./test.csv")

testing_passengers = []
testing_id = []
testing_survived = []
testing_results = []

for i in range(len(testing_data)):
	passenger = []
	# Scale per value
	for ti,t in enumerate(trainable):
		value = np.array(np.float(testing_data[t][i])).reshape(1, -1) if not np.isnan(testing_data[t][i]) else [[0]]
		passenger.append(scalers[ti].transform(value)[0][0])
	passenger.append(1 if testing_data["Sex"][i] == "male" else 0) # Gender
	testing_passengers.append(passenger)
	testing_id.append(testing_data["PassengerId"][i])

testing_passengers = np.array(testing_passengers)
testing_id = np.array(testing_id)

for i,v in enumerate(testing_passengers):
	testing_survived.append(predict(v))
	testing_results.append([testing_id[i], testing_survived[i]])

testing_survived = np.array(testing_survived, dtype=int)
testing_results = np.array(testing_results, dtype=int)

# To CSV
pd.DataFrame(testing_results).to_csv("results.csv", index=None, header=["PassengerId", "Survived"])