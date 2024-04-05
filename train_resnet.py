import os
import numpy as np
import datetime as dt
from resnet3d import Resnet3DBuilder

# Access the JOINT_TYPE environment variable, default to 'hip'
joint = os.getenv('JOINT_TYPE')

print(f"Training {joint} data")

# Load data
x_train = np.load(f"data/arrays/x_train_{joint}.npy")
x_val = np.load(f"data/arrays/x_val_{joint}.npy")
x_test = np.load(f"data/arrays/x_test_{joint}.npy")
y_train = np.load(f"data/arrays/y_train_{joint}.npy")
y_val = np.load(f"data/arrays/y_val_{joint}.npy")
y_test = np.load(f"data/arrays/y_test_{joint}.npy")

# Reshape data to add channel dimension
x_train = x_train.reshape(*x_train.shape, 1)
x_val = x_val.reshape(*x_val.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

# Build model
input_shape = (x_train.shape[1:])
nb_classes = y_train.shape[1]

model = Resnet3DBuilder.build_resnet_50(input_shape, nb_classes)
model.compile(optimizer="adam", 
              loss="mean_squared_error", 
              metrics=["mae"])

# Train model
start_time = dt.datetime.today()
print("Train start time: " + str(start_time))

history = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val),
                    batch_size=1, 
                    epochs=100, 
                    verbose=1)

finish_time = dt.datetime.today()
print("Train finish time: " + str(finish_time))
print("Training time: " + str(finish_time - start_time))

# Save model
file_path = f"data/models/resnet_{joint}"
json_string = model.to_json()
fj = file_path + ".json"
fh = file_path + ".h5"
open(fj, "w").write(json_string)
model.save_weights(fh)
print(f"Model saved to: " + file_path)

# Test model on validation dataset
print("Validation dataset")
start_time = dt.datetime.today()
print("Prediction start time: " + str(start_time))

y_pred = model.predict(x_val, batch_size=1, verbose=1)

finish_time = dt.datetime.today()
print("Prediction finish time: " + str(finish_time))
print("Prediction time: " + str(finish_time - start_time))

np.save(f"data/arrays/y_pred_val_{joint}.npy", y_pred)

# Test model on test dataset
print("Test dataset")
start_time = dt.datetime.today()
print("Prediction start time: " + str(start_time))

y_pred = model.predict(x_test, batch_size=1, verbose=1)

finish_time = dt.datetime.today()
print("Prediction finish time: " + str(finish_time))
print("Prediction time: " + str(finish_time - start_time))

np.save(f"data/arrays/y_pred_test_{joint}.npy", y_pred)