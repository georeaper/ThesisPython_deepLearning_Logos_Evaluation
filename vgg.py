import six.moves.cPickle as pickle
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 200 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons)




train_file=open('C:/Users/MasterPC/pythonprogs/train225x225_6000_edge.pickle', 'rb')
data=pickle.load(train_file)
x,y=data
x_train=np.array(x)
y_train=np.array(y)
train_file.close()
print(x_train[1].shape)
#print("Train file Loaded....Loading Test file")
test_file=open('C:/Users/MasterPC/pythonprogs/test225x225_200_edge.pickle', 'rb')
data=pickle.load(test_file)
x,y=data
x_test=np.array(x)
y_test=np.array(y)
test_file.close()

print("training Data shapes : ",x_train.shape,y_train.shape)
print("test Data shapes : ",x_test.shape,y_test.shape)

classes=np.unique(y_train)
nClasses=len(classes)
print("Total number of outputs : ", nClasses)
print('Output classes : ',classes)

train_X=x_train.reshape(-1,225,225,1)

test_X=x_test.reshape(-1,225,225,1)

print(train_X.shape)

print(test_X.shape)

train_X=train_X.astype('float32')
test_X=test_X.astype('float32')
train_X=train_X/255
test_X=test_X/255

train_Y_one_hot=to_categorical(y_train)
test_Y_one_hot=to_categorical(y_test)

print('Original Label: ',y_train[0])
print('After conversion: ',train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

#batch_size = 10
#epochs = 3
#num_classes = 2

#inp = Input(shape=(height, width, depth))
conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size),input_shape=(225,225,1), padding='same', activation='relu')
conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')
conv_4 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

#model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='binary_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation
model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

#fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
with tf.device('/gpu:0'):
	fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(valid_X, valid_label))

test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=2)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
print(test_eval)