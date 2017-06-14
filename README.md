A collection of basic ML examples, all written in Python 3 for TensorFlow/Keras.

An excerpt of output from one run of conv.py, edited for clarity:

```
name: GeForce GTX 1080
major: 6 minor: 1 memoryClockRate (GHz) 1.7335
pciBusID 0000:01:00.0
Total memory: 8.00GiB
Free memory: 6.63GiB

Epoch 1/20
4s - loss: 4.1814 - acc: 0.0881 - val_loss: 3.7754 - val_acc: 0.1510
Epoch 2/20
3s - loss: 3.5506 - acc: 0.1950 - val_loss: 3.3360 - val_acc: 0.2430
Epoch 3/20
3s - loss: 3.2356 - acc: 0.2544 - val_loss: 3.0055 - val_acc: 0.3090
Epoch 4/20
3s - loss: 3.0222 - acc: 0.2921 - val_loss: 2.9114 - val_acc: 0.3270
Epoch 5/20
3s - loss: 2.8665 - acc: 0.3194 - val_loss: 2.6631 - val_acc: 0.3790
Epoch 6/20
3s - loss: 2.7535 - acc: 0.3402 - val_loss: 2.6085 - val_acc: 0.3880
Epoch 7/20
3s - loss: 2.6621 - acc: 0.3565 - val_loss: 2.5497 - val_acc: 0.3910
Epoch 8/20
3s - loss: 2.5829 - acc: 0.3689 - val_loss: 2.5327 - val_acc: 0.3790
Epoch 9/20
3s - loss: 2.5297 - acc: 0.3774 - val_loss: 2.3877 - val_acc: 0.4150
Epoch 10/20
3s - loss: 2.4767 - acc: 0.3842 - val_loss: 2.3483 - val_acc: 0.4270
Epoch 11/20
3s - loss: 2.4232 - acc: 0.3970 - val_loss: 2.3352 - val_acc: 0.4230
Epoch 12/20
3s - loss: 2.3939 - acc: 0.3988 - val_loss: 2.3096 - val_acc: 0.4370
Epoch 13/20
3s - loss: 2.3572 - acc: 0.4045 - val_loss: 2.2810 - val_acc: 0.4350
Epoch 14/20
3s - loss: 2.3252 - acc: 0.4093 - val_loss: 2.3074 - val_acc: 0.4260
Epoch 15/20
3s - loss: 2.2907 - acc: 0.4176 - val_loss: 2.2557 - val_acc: 0.4500
Epoch 16/20
3s - loss: 2.2647 - acc: 0.4238 - val_loss: 2.2957 - val_acc: 0.4300
Epoch 17/20
3s - loss: 2.2428 - acc: 0.4260 - val_loss: 2.2874 - val_acc: 0.4250
Epoch 18/20
3s - loss: 2.2317 - acc: 0.4287 - val_loss: 2.2027 - val_acc: 0.4550
Epoch 19/20
3s - loss: 2.2069 - acc: 0.4350 - val_loss: 2.2332 - val_acc: 0.4360
Epoch 20/20
3s - loss: 2.1872 - acc: 0.4352 - val_loss: 2.2290 - val_acc: 0.4480
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1 (Conv2D)               (None, 32, 32, 32)        896
_________________________________________________________________
activation_1 (Activation)    (None, 32, 32, 32)        0
_________________________________________________________________
pool1 (MaxPooling2D)         (None, 16, 16, 32)        0
_________________________________________________________________
conv2 (Conv2D)               (None, 16, 16, 32)        9248
_________________________________________________________________
activation_2 (Activation)    (None, 16, 16, 32)        0
_________________________________________________________________
pool2 (MaxPooling2D)         (None, 8, 8, 32)          0
_________________________________________________________________
conv3 (Conv2D)               (None, 8, 8, 32)          9248
_________________________________________________________________
activation_3 (Activation)    (None, 8, 8, 32)          0
_________________________________________________________________
pool3 (MaxPooling2D)         (None, 4, 4, 32)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense1 (Dense)               (None, 100)               51300
_________________________________________________________________
batch_normalization_1 (Batch (None, 100)               400
_________________________________________________________________
activation_4 (Activation)    (None, 100)               0
=================================================================
Total params: 71,092
Trainable params: 70,892
Non-trainable params: 200
_________________________________________________________________
None
score:  [2.2557770912170412, 0.43209999999999998]
metrics:  ['loss', 'acc']
```
