from data_reader import get_smile_intVectors_and_targets, get_data
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from keras.utils import to_categorical
# fix random seed for reproducibility


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def parse_data():
    d_train, d_labels = get_smile_intVectors_and_targets(get_data())
    x_train = []
    x_labels =[]
    for k in d_train:
        x_train.append(d_train[k])
        x_labels.append(d_labels[k])
    return np.array(x_train), np.array([i[0] if len(i) == 1 else i for i in x_labels])

def get_target_data(target_index, x_train, x_labels):
    x = []
    l = []
    for label in range(len(x_labels)):
        if x_labels[label][target_index] != "":
            l.append(x_labels[label][target_index])
            x.append(x_train[label])
    return np.array(x), np.array(l)

def build_lstm_model(max_len, x_train):
    embedding_length = 100
    model = Sequential()
    model.add(Embedding(len(x_train), embedding_length,
                        input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3,
                     padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(LSTM(100))
    model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])
    return model


def dpt_model(max_len, x_train):
    embedding_length = 100
    dpt_model = keras.models.Sequential([
        Embedding(len(x_train), embedding_length,input_length=max_len),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    dpt_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])
    return dpt_model

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

def build_network(target_index):
    x_train, x_labels = parse_data()
    x_train, x_labels = get_target_data(target_index, x_train, x_labels)
    max_len = len(max(x_train, key=len))

    x_train = sequence.pad_sequences(
        x_train, maxlen=max_len)

    spl = (len(x_train)*80)//100
    x_train, x_test = np.array(x_train[0:spl]), np.array(x_train[spl:])
    y_train, y_test= np.array(x_labels[0:spl]), np.array(x_labels[spl:])

    #model
    model = build_lstm_model(max_len, x_train)

    hist = AccuracyHistory()
    print(model.summary())
    history = model.fit(x_train,
                y_train,
                validation_data=(x_test, y_test),
                epochs=10,
                batch_size=128,
                verbose=1,)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

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

for i in range(0,1):
    print("TARGET: %d" %(i+1))
    build_network(i)


