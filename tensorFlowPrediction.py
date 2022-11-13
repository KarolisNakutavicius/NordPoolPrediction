import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10) # Output
])
# print(model.summary())

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics) # Confirugring model for training
# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# evaluate
results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
print("test loss, test acc:", results)

# predictions

probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model(x_test)
print(predictions[0])
label0 = np.argmax(predictions[0])
print(label0)

plt.imshow(x_test[0])
plt.show()





# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(x_train[i], cmap='gray')
# plt.show()


# print(x_train.shape, y_train.shape)
