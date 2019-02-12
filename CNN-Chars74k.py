'''
Step 0 : Importing the libraries
Step 1 : Locating a patha to the stored file
Step 2: Preprocessing --> Grayscaling/Resizing/Format changing (Skipped)
Step 3: Create an array that has all the images flattened out. Array is type <ndarray>
Step 4: Define the label for the images. Array is of <ndarray> type. Part of supervised learning.
Step 5: Let's shuffle the order of images and labels. Corresponding images would be shuffled with their labels.
Step 6:
Step 7:

'''

import keras
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import MaxPooling2D, Convolution2D

import numpy as np
from keras.utils import np_utils

from numpy import array
from numpy import size
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os


"%matplotlib inline"

path1 = 'D:\Coding\OpenCV\Sudoku\All'
path2 = 'D:\Coding\OpenCV\Sudoku\All_Proc'

img_rows= img_cols = 128

listing = os.listdir(path1)
num_samples = size(listing)
print(num_samples)

# for file in listing:
#     Image.open(path1 + "\\" + file)

image1 = (Image.open(path1 + "\\" + listing[0]))
image2 = array(Image.open(path1 + "\\" + listing[0])).flatten()

imgmatrix = array([array(Image.open(path1 + '\\' + img)).flatten()for img in listing],"f")

label = np.zeros((num_samples,),dtype=int)
label.shape

listingnd = np.array(listing)

label[0:1016] = 0
label[1016:2032] = 1
label[2032:3048] = 2
label[3048:4064] = 3
label[4064:5080] = 4
label[5080:6096] = 5
label[6096:7112] = 6
label[7112:8128] = 7
label[8128:9144] = 8
label[9144:10160] = 9

data,Label = shuffle(imgmatrix,label,random_state = 2)
temp = [data,Label]

print(temp[0].shape)
print(temp[1].shape)

#batch_size to train
batch_size = 256
# number of output classes
nb_classes = 10
# number of epochs to train
nb_epoch = 1


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

X = temp[0]
y = temp[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4)
X_train.shape
y_test.shape


X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,border_mode='valid',input_shape=(img_rows, img_cols,1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))

from keras.models import load_model
model.save("My_model.h5")