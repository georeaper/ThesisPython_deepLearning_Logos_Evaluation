import six.moves.cPickle as pickle
import numpy
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from functions_general import give_arguments as gvarg
import tensorflow as tf


images=open('C:/Users/MasterPC/pythonprogs/images_logos.pickle', 'rb')
images=pickle.load(images)

data=open('C:/Users/MasterPC/pythonprogs/y_logos.pickle', 'rb')

y_logos=pickle.load(data)

indices,active,comple_x,depth=y_logos


train_images=images[0:182]
train_images=numpy.array(train_images)
test_images=images[183:208]
test_images=numpy.array(test_images)


#print(train_images.shape)
#print(len(comple_x))
Ccomp=[]
for i in range(0,208):
	mean_agg=int(comple_x[i][0])+int(comple_x[i][1])+int(comple_x[i][2])
	mean_c=mean_agg/3
	#print(int(mean_c))
	Ccomp.append(int(mean_c))
#print(len(images))
Ccomp=numpy.array(Ccomp)
#print(Ccomp)

Cact=[]
for i in range(0,208):
	mean_agg=int(active[i][0])+int(active[i][1])+int(active[i][2])
	mean_c=mean_agg/3
	#print(int(mean_c))
	Cact.append(int(mean_c))
train_act_y=Cact[0:182]
test_act_y=Cact[183:209]
#print(test_act_y)
train_act_y=numpy.array(train_act_y)
test_act_y=numpy.array(test_act_y)
Cdep=[]
for i in range(0,208):
	mean_agg=int(depth[i][0])+int(depth[i][1])+int(depth[i][2])
	mean_c=mean_agg/3
	#print(int(mean_c))
	Cdep.append(int(mean_c))

Cdep=numpy.array(Cdep)

# train_X=train_images.reshape(-1,225,225,1)
# print("done")
train_Y_one_hot=to_categorical(train_act_y)
test_Y_one_hot=to_categorical(test_act_y)
classes=numpy.unique(train_act_y)
nClasses=len(classes)
print("Total number of outputs : ", nClasses)
print('Output classes : ',classes)
print(train_images.shape)

train_X,valid_X,train_label,valid_label = train_test_split(train_images, train_Y_one_hot, test_size=0.2, random_state=13)
#print(train_X)
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Convolution2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dense, Dropout, Flatten

input_shape=(None, None,3)
m = Sequential()
m.add(Convolution2D(8, 3, 3, input_shape=input_shape))
m.add(keras.layers.GlobalMaxPooling2D(data_format=None))
#m.add(Flatten())
m.add(Dense(5, activation='sigmoid'))
m.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
m.summary()
m_train = m.fit(train_X, train_label, batch_size=10,epochs=10,verbose=1,validation_data=(valid_X, valid_label))

