import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

text = open("trump.txt").read()
text = text.lower()

#Mapping  -  We assign an arbitrary number to each element in the text for identification - this facilitates the training process as machines
#understand numbers far better than words

#a dictionary is created where letters are enumerated to uniquely identify words
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}


x = []
y = []
length = len(text)
seq_length = 100

for i in range(0,length-seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    x.append([char_to_n[char] for char in sequence])
    y.append(char_to_n[label])

x_mod = np.reshape(x, (len(x),seq_length, 1))
x_mod = x_mod/float(len(characters))
y_mod = np_utils.to_categorical(y)


model = Sequential()
model.add(LSTM(400, input_shape=(x_mod.shape[1], x_mod.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(y_mod.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.load_weights('trumpweights_2.h5')


#This is where the text generation occurs


string_mapped = x[6417]
full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(200):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt=""
for char in full_string:
    txt = txt+char
print(txt)