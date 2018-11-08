import utils
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
    model.add(Dense(12, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy', 'binary_crossentropy'])
    return model


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

def build_model():
    vectors = utils.get_vectors_from_smiles()
    targets = utils.get_augmented_targets()

    x_train, x_labels = vectors, targets

    max_len = len(max(x_train, key=len))
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)

    spl = (len(x_train)*80)//100
    x_train, x_test = np.array(x_train[0:spl]), np.array(x_train[spl:])
    y_train, y_test = np.array(x_labels[0:spl]), np.array(x_labels[spl:])

    #model
    model = build_lstm_model(max_len, x_train)

    print(model.summary())
    history = model.fit(x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=10,
                        batch_size=256,
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

build_model()
