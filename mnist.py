from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import Input, Model
import keras.backend as K
import matplotlib.pylab as plt
import numpy as np

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_x, img_y = 28, 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np.array([1,0,0,0,0,1,1,0])
print(y_train)

y_train = keras.utils.to_categorical(y_train, 2)

print(y_train)