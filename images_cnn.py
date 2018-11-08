from data_reader import get_data
import cv2
import keras
import numpy as np
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras import Input, Model
import matplotlib.pylab as plt


d = get_data()
index = 0
images = []
for i in d["smiles"]:
    img = cv2.imread("images/" + str(index) + ".png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(gray)
    index += 1

target1_for_training = []

for t in d["target1"]:
    if t != "":
        target1_for_training.append(int(t))
    else:
        target1_for_training.append(0)

target1_for_training = target1_for_training[0:1000]
img_x = 300
img_y = 300
x_train = np.array(images)[0:1000]
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)
# convert the data to the right type
x_train = x_train.astype('float32')
x_train /= 255

y_train = keras.utils.to_categorical(np.array(target1_for_training), 2)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()
print(model.summary())
model.fit(x_train, y_train,
          batch_size=128,
          epochs=3,
          verbose=1,
          validation_split=0.2,
          callbacks=[history])

# Save output to CSV
# np.savetxt("foo.csv", K.eval(output) , delimiter=",")

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()
