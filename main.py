import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

print(X_train.shape)
print(X_train.dtype)

X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

class_names = ["Koszulka", "Spodnie", "Sweter", "Sukienka", "Płaszcz", "Sandał", "Koszula", "Tenisówka", "Torebka", "Trzewik"]

print(class_names[y_train[0]])

tf.random.set_seed(42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation="relu"))
model.add(tf.keras.layers.Dense(100, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoka", style=["r--", "r--.", "b-", "b-*"])
plt.show()

sample_image = X_test[6]
plt.imshow(sample_image, cmap="gray")
plt.show()

sample_image = sample_image.reshape(1, 28, 28)
predictions = model.predict(sample_image)
predicted_class = np.argmax(predictions)

print(f"Przewidziana klasa: {class_names[predicted_class]}")