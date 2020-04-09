# Basic classifier for text to see if reviews are good or bad

import tensorflow as tf
from tensorflow import keras
import numpy as np 


# henter datasettet til imdb
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000)

# definerer word_index som gjør om ord til tall
word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# definerer reverse_word_index for å kunne gjøre tall til ord
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# definerer hva som er train data og hva som er test data, og setter max lengde på review til 250 ord. 
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# for å kunne vise teksten i plaintekst. F.eks. print(decode_review(test_data))
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#model
# sekvensiell modell
model = keras.Sequential()
# for å kunne beskrive ord matematisk brukes vektorerregning med 16 koeffisienter, hvor totalt antall vektorer i språket er satt til 100000. 
model.add(keras.layers.Embedding(10000, 16)) 
# Er en Averager som finner gjennomsnittet av en vektor. Hver vektor har 16 koeffisenter
model.add(keras.layers.GlobalAveragePooling1D())
 # hidden layer, 16 noder ettersom 
model.add(keras.layers.Dense(16, activation="relu"))
# output layer. Bruker sigmoid til å få en verdi mellom 0 og 1
model.add(keras.layers.Dense(1, activation="sigmoid")) 

model.summary()
# man må comple modellen for å få den til å virke
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# deler train data i to deler
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# definerer fitModel til å trene 40 ganger
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

test_review = test_data[1]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[1]))
print("Actual: " + str(test_labels[1]))


