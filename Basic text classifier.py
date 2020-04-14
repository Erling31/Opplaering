# Basic classifier for text to see if reviews are good or bad

import tensorflow as tf
from tensorflow import keras
import numpy as np 


# henter datasettet til imdb
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 88000)

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

'''
#model
# sekvensiell modell
model = keras.Sequential()
# for å kunne beskrive ord matematisk brukes vektorerregning med 16 koeffisienter, hvor totalt antall vektorer som kan beskrive språket er satt til 100000. 
# F.eks. kan ordet "bukse" ha verdien 7, som kan beskrives som vektor med 16 koeffisienter.
# Embedded layer tar imput layer (selve ordet) og gjør det om til en vektor, og sender den videre til f.eks. en average layer
model.add(keras.layers.Embedding(88000, 16)) 
# Er en Averager som finner gjennomsnittet av en vektor. Hver vektor har 16 koeffisenter
model.add(keras.layers.GlobalAveragePooling1D())
 # hidden layer, 16 noder. Den som klassifiserer. 
model.add(keras.layers.Dense(16, activation="relu"))
# output layer. Bruker sigmoid til å få en verdi mellom 0 og 1
model.add(keras.layers.Dense(1, activation="sigmoid")) 

model.summary()
# man må comple modellen for å få den til å virke
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# deler train data i to deler. Fra importen er det 25000 reviews. Her 
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# definerer fitModel til å trene 40 ganger
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

model.save("model.h5")'''

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

model = keras.models.load_model("model.h5")

with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


'''
test_review = test_data[1]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[1]))
print("Actual: " + str(test_labels[1]))
'''



