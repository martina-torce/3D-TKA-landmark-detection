import datetime as dt
import numpy as np
import pickle
from model_resnet import Resnet3DBuilder
from tensorflow.keras.callbacks import Callback

# Load data
x_train = np.load("data/x_trainval_hip.npy")
y_train = np.load("data/y_trainval_hip.npy")

# Reshape data
ir = x_train.shape[1]  # rows
ic = x_train.shape[2]  # cols
ih = x_train.shape[3]  # slices
x_train = x_train.reshape(x_train.shape[0], ir, ic, ih, 1)

# Build model
input_shape = (ir, ic, ih, 1)
nb_classes = y_train.shape[1]

model = Resnet3DBuilder.build_resnet_50(input_shape, nb_classes)
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Custom callback
# output intermediate steps!!

# Train model
start_time = dt.datetime.today()
print("Train start time: " + str(start_time))

history = model.fit(x_train, y_train, batch_size=1, epochs=500, verbose=1)

finish_time = dt.datetime.today()
print("Train finish time: " + str(finish_time))
print("Training time: " + str(finish_time - start_time))

# Save model
file_path = "data/models/resnet_hip"
json_string = model.to_json()
fj = file_path + ".json"
fh = file_path + ".h5"
open(fj, "w").write(json_string)
model.save_weights(fh)
print("Model saved to: " + file_path)

# Test model
y_test_predictions = model.predict(x_train[:2])
y_test_predictions[0].shape
print(y_test_predictions[0])
