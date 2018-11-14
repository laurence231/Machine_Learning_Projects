import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


#We open the text in here

text = open("trump.txt").read()
text = text.lower()


#Mapping  -  We assign an arbitrary number to each element in the text for identification - this facilitates the training process as machines
#understand numbers far better than words

#a dictionary is created where letters are enumerated to uniquely identify words
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

#Data preprocessing - transform data into relatable format

x = []
y = []
length = len(text)
seq_length = 100

for i in range(0,length-seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    x.append([char_to_n[char] for char in sequence])
    y.append(char_to_n[label])

#X is training array and y is target array
#seq length is the length of character sequence we want to consider before predicting characters
#the for loop iterates over the entire sequence of text and creates sequences, stores in x, true values stored in y.

#we now format the arrays

x_mod = np.reshape(x, (len(x),seq_length, 1))
x_mod = x_mod/float(len(characters))
y_mod = np_utils.to_categorical(y)

#here the model is formed using keras

model = Sequential()
model.add(LSTM(400, input_shape=(x_mod.shape[1], x_mod.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400))
model.add(Dropout(0.2))
model.add(Dense(y_mod.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x_mod, y_mod, epochs=20, batch_size=50)

model.save_weights('trumpweights_2.h5')
# model.load_weights('weights.h5')
#
#
# #This is where the text generation occurs
# string_mapped = x[99]
# # generating characters
# for i in range(seq_length):
#     x = np.reshape(string_mapped,(1,len(string_mapped), 1))
#     x = x / float(len(characters))
#     pred_index = np.argmax(model.predict(x, verbose=0))
#     seq = [n_to_char[value] for value in string_mapped]
#     string_mapped.append(pred_index)
#     string_mapped = string_mapped[1:len(string_mapped)]
#
# txt=""
# for char in full_string:
#     txt = txt+char
# print(txt)