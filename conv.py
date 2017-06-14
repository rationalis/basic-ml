import h5py
import numpy as np
np.random.seed(1337) # reproducibility
# There still appears to be variation between runs.

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical

'''

A simple convolutional neural network, with 3 convolution layers of 32 filters
each and a single dense layer, for classifying CIFAR-100. ~71K parameters,
obtains ~43% test accuracy after 20 epochs. Takes ~3s/epoch to train on a GTX
1080.

'''

epochs = 20
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

Y_train, Y_test = to_categorical(y_train, 100), to_categorical(y_test, 100)

model = Sequential([
  Conv2D(32, (3,3), padding='same', input_shape=(32,32,3), name='conv1'),
  Activation('relu'),
  MaxPooling2D(pool_size=(2,2), name='pool1'),

  Conv2D(32, (3,3), padding='same', name='conv2'),
  Activation('relu'),
  MaxPooling2D(pool_size=(2,2), name='pool2'),

  Conv2D(32, (3,3), padding='same', name='conv3'),
  Activation('relu'),
  MaxPooling2D(pool_size=(2,2), name='pool3'),

  Flatten(),
  Dropout(0.4),

  Dense(100, name='dense1'),
  BatchNormalization(),
  Activation('softmax')])

model.compile(optimizer='nadam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def standardize(x):
  x_prime = (x - np.mean(x, axis=(0,1), keepdims=True))
  x_prime /= (np.std(x, axis=(0,1), keepdims=True) + 1e-7)
  return x_prime

X_train_gcn = np.empty(X_train.shape, dtype=np.float32)
for ind, x in enumerate(X_train):
  X_train_gcn[ind] = standardize(x)

X_test_gcn = np.empty(X_test.shape, dtype=np.float32)
for ind, x in enumerate(X_test):
  X_test_gcn[ind] = standardize(x)

tensorboard = TensorBoard(histogram_freq=1)

hist = model.fit(X_train_gcn, Y_train, epochs=epochs,
                 batch_size=400, verbose=1,
                 validation_data=(X_test_gcn[:1000],Y_test[:1000]),
                 callbacks=[tensorboard])

score = model.evaluate(X_test_gcn, Y_test, verbose=1)

model.save_weights('model_conv.h5')

print('')
print(model.summary())
print('score: ', score)
print('metrics: ', model.metrics_names)
