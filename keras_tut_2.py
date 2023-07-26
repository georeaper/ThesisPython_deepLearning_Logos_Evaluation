import six.moves.cPickle as pickle
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from functions_general import give_arguments as gvarg
import tensorflow as tf

# print(sys.argv)
# arguments=gvarg(int(sys.argv[1]))

# batch_size=int(arguments[0])
# epochs=int(arguments[1])
# conv1=int(arguments[2])
# ker1=int(arguments[3])
# conv2=int(arguments[4])
# ker2=int(arguments[5])
# conv3=conv2+15
# ker3=ker2
# dense1=int(arguments[6])
# print(batch_size,epochs)
# print(conv1,ker1)
# print(conv2,ker2)
# print(conv3,ker3)
# print(dense1)




train_file=open('C:/Users/MasterPC/pythonprogs/train225x225_10000_edge.pickle', 'rb')
data=pickle.load(train_file)
x,y=data
x_train=np.array(x)
y_train=np.array(y)
train_file.close()
print(x_train[1].shape)
#print("Train file Loaded....Loading Test file")
#test_file=open('C:/Users/MasterPC/pythonprogs/test225x225_200_edge.pickle', 'rb')
#data=pickle.load(test_file)
#x,y=data
#x_test=np.array(x)
#y_test=np.array(y)
#test_file.close()

print("training Data shapes : ",x_train.shape,y_train.shape)
#print("test Data shapes : ",x_test.shape,y_test.shape)

classes=np.unique(y_train)
nClasses=len(classes)
print("Total number of outputs : ", nClasses)
print('Output classes : ',classes)

train_X=x_train.reshape(-1,225,225,1)

#test_X=x_test.reshape(-1,225,225,1)

print(train_X.shape)

#print(test_X.shape)

train_X=train_X.astype('float32')
#test_X=test_X.astype('float32')
train_X=train_X/255
#test_X=test_X/255

train_Y_one_hot=to_categorical(y_train)
#test_Y_one_hot=to_categorical(y_test)

print('Original Label: ',y_train[0])
print('After conversion: ',train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

#batch_size = 10
#epochs = 3
num_classes = 2


fashion_model = Sequential()
fashion_model.add(Conv2D(64, kernel_size=(3,3),activation='linear',input_shape=(225,225,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3,3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(256, (3,3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(256, activation='sigmoid'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='sigmoid'))

fashion_model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
fashion_model.summary()

#fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
with tf.device('/gpu:0'):
	fashion_train = fashion_model.fit(train_X, train_label, batch_size=10,epochs=10,verbose=2,validation_data=(valid_X, valid_label))

#test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])