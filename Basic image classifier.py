# Test av Github

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# importer data fashin_mnist
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# navn på de ulike klesartiklene
class_names = ['T-shirt', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal', 
                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Deler bildene slik at de blir mindre
train_images = train_images/255.0
test_images = test_images/255.0

# Modellen: to layers. imput forteller at bildene er str. 28 x 28, hidden layer er på 128 noder og output layer er på 10 noder. 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
metrics=["accuracy"])
# trener modellen til å gjennkjenne images basert på labelst. Epochs sier noe om hvor mange ganger den skal trenes. 
model.fit(train_images, train_labels, epochs=5)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i] , cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()




