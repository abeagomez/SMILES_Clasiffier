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
np.random.seed(7)

smiles = utils.get_smiles_as_vectors()
original_targets =utils.get_filled_targets()
target_index = 0
target = [target[target_index] for target in original_targets]
pos_s = utils.get_positive_samples(target_index, smiles, original_targets)
x_train, y = utils.over_sampling_data_set(0,pos_s, smiles, target)

def training_data(x_train, y):
    max_len = len(max(x_train, key=len))
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x = [[[float(i)] for i in x] for x in x_train]
    x = np.array(x)
    x = x.reshape(x.shape[0], max_len, 1)
    y = np.array(y)
    y = keras.utils.to_categorical(y, 2)
    spl = (len(x_train)*80)//100
    x_train, x_test = np.array(x[0:spl]), np.array(x[spl:])
    y_train, y_test = np.array(y[0:spl]), np.array(y[spl:])
    return x_train, x_test, y_train, y_test

def build_model():
    print("Build model")
    model = Sequential()
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, input_dim=1))
    model.add(Dense(2, activation="sigmoid"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

def show_metrics(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

x_train, x_test, y_train, y_test = training_data(x_train, y)
print("Train")
model = build_model()
history= model.fit(x_train, y_train,
                epochs=1,
                validation_data=(x_test, y_test),
                verbose=1)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

show_metrics(history)
