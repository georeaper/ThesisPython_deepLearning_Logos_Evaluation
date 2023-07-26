import six.moves.cPickle as pickle
import numpy
from keras.utils import to_categorical
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from functions_general import give_arguments as gvarg
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
from sklearn.utils import class_weight
# images=open('C:/Users/MasterPC/pythonprogs/images_logos21.pickle', 'rb')
# images=pickle.load(images)

# data=open('C:/Users/MasterPC/pythonprogs/y_logos.pickle', 'rb')

# y_logos=pickle.load(data)

# indices,active,comple_x,depth=y_logos
# #print(active)


# train_images=images[0:208]
# train_images=numpy.array(train_images)

# test_images=images[206:208]
# test_images=numpy.array(test_images)


#print(train_images.shape)
#print(len(comple_x))
####################
####################
# Ccomp=[]
# for i in range(0,208):
# 	mean_agg=int(comple_x[i][0])+int(comple_x[i][1])+int(comple_x[i][2])
# 	mean_c=mean_agg/3
# 	#print(int(mean_c))
# 	Ccomp.append(int(mean_c))
# #print(len(images))
# Ccomp=numpy.array(Ccomp)
# #print(Ccomp)
####################
####################

# Cact=[]
# for i in range(0,208):
# 	mean_agg=int(active[i][0])+int(active[i][1])+int(active[i][2])
# 	mean_c=mean_agg/3
# 	#print(int(mean_c))
# 	Cact.append(round(mean_c))
###################
###################
#####test me augmented logos#####
data=open('C:/Users/MasterPC/pythonprogs/augmented_logos.pickle', 'rb')

logo_data=pickle.load(data)	
x,y=logo_data

train_act_y=y


x=Counter(train_act_y)
print(x)

#test_act_y=Cact[206:209]
#print(test_act_y)
train_act_y=numpy.array(train_act_y)
#test_act_y=numpy.array(test_act_y)
class_weight = class_weight.compute_class_weight('balanced',
                                                 numpy.unique(train_act_y),
                                                 train_act_y)
print(class_weight)
#################
################
# Cdep=[]
# for i in range(0,208):
# 	mean_agg=int(depth[i][0])+int(depth[i][1])+int(depth[i][2])
# 	mean_c=mean_agg/3
# 	#print(int(mean_c))
# 	Cdep.append(int(mean_c))

# Cdep=numpy.array(Cdep)

# train_X=train_images.reshape(-1,225,225,1)
#########################
##########################
# print("done")
train_Y_one_hot=to_categorical(train_act_y)
#test_Y_one_hot=to_categorical(test_act_y)
#print(train_Y_one_hot)
classes=numpy.unique(train_act_y)
nClasses=len(classes)
print("Total number of outputs : ", nClasses)
print('Output classes : ',classes)
print(train_images.shape)

train_X=x.reshape(-1,225,225,3)

#test_X=x_test.reshape(-1,225,225,1)

#print(train_X.shape)

#print(test_X.shape)

train_X,valid_X,train_label,valid_label = train_test_split(train_images, train_Y_one_hot, test_size=0.2, random_state=13)


#print(train_X)
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
#from keras.utils import plot_model


fashion_model = Sequential()
fashion_model.add(Conv2D(64, kernel_size=(2,2),activation='relu',input_shape=(225,225,3),padding='same'))
fashion_model.add(Conv2D(64, (2,2), activation='relu',padding='same'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(BatchNormalization())
#fashion_model.add(Conv2D(32, (2,2), activation='relu',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
#fashion_model.add(MaxPooling2D((2, 2),padding='same'))

fashion_model.add(Conv2D(128, (2,2), activation='relu',padding='same'))
fashion_model.add(Conv2D(128, (2,2), activation='relu',padding='same'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(BatchNormalization())


fashion_model.add(Conv2D(256, (2,2), activation='relu',padding='same'))
fashion_model.add(Conv2D(256, (2,2), activation='relu',padding='same'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(BatchNormalization())
#fashion_model.add(Conv2D(64, (2,2), activation='relu',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
# #fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# fashion_model.add(Conv2D(32, (2,2), activation='relu',padding='same'))
# fashion_model.add(Conv2D(32, (2,2), activation='relu',padding='same'))
# fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# fashion_model.add(BatchNormalization())

fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(Dense(128, activation='relu'))
#fashion_model.add(Dropout(0.2))                  
fashion_model.add(Dense(7, activation='softmax'))
fashion_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
fashion_model.summary()
with tf.device('/gpu:0'):
	#history = fashion_model.fit(train_X, train_label, batch_size=20,epochs=20,verbose=1,validation_data=(valid_X, valid_label))
	history = fashion_model.fit(train_X, train_label,class_weight=class_weight, batch_size=20,epochs=3,verbose=1)

# history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
ynew = fashion_model.predict_classes(valid_X)
for i in range(len(valid_X)):
	print("X=%s, Predicted=%s" % (valid_label[i], ynew[i]))
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()