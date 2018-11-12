import utils
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from keras.utils import to_categorical
import tensorflow as tf
import keras.backend as K
# fix random seed for reproducibility

max_features = 4
y = np.array([1,2,3,2,3,1,0])
y2 = np.zeros((y.shape[0],max_features), dtype=np.float32)
y2[np.arange(y.shape[0]), y] = 1.0
#print(y2)

x_train = utils.get_smiles_as_vectors()
max_len = len(max(x_train, key=len))
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x = [[[float(i)] for i in x] for x in x_train]
x = np.array(x)
x = x.reshape(x.shape[0], max_len, 1)
targets = utils.get_filled_targets()
y = np.array(targets)

print("Build model")
model = Sequential()
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_dim=1))
model.add(Dense(12, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])

print("Train")
model.fit(x,y,epochs=10, validation_split=0.2, verbose=1)
pred = model.predict(x)
predict_classes = np.argmax(pred,axis=1)
print("Predicted classes: {}", predict_classes)
print("Expected classes: {}", predict_classes)
