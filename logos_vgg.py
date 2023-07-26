from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions_general import logos_pkl,train_valid_augmented_data
from keras import optimizers
#from deep_logos_augmented import train_valid_augmented_data
from sklearn.utils import class_weight
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback
#from numpy.random import randint
from numpy import argmax

class EarlyStoppingByAcc(Callback):
    def __init__(self, monitor='acc', value=0.92, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        print(current)
        current=float(current)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)
            exit()

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

train_X, valid_X, train_label, valid_label, class_weight1=train_valid_augmented_data('augmented_class1_logos_complexity.pickle')

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=(225,225,3),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dense(2048, activation='relu', name='fc2')(x)
x = Dense(7, activation='softmax', name='predictions')(x)

#Create your own model 
sgd = optimizers.SGD(lr=0.0001)
my_model = Model(input=input, output=x)
my_model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()


#Then training with your data ! 
#train_X,valid_X,train_label,valid_label=logos_pkl()


# with tf.device('/gpu:0'):
# 	history = my_model.fit(train_X, train_label,class_weight=class_weight, batch_size=10,epochs=15,verbose=1)
history = my_model.fit(train_X, train_label,class_weight=class_weight1, batch_size=20,epochs=10,verbose=2,callbacks=[EarlyStoppingByAcc()])


preds = my_model.predict(valid_X)
# cm = confusion_matrix(valid_label, preds)
# sns.heatmap(cm,annot=True,cmap="Set2")

for i in range(len(valid_X)):
	print("X=%s, Predicted=%s" % (valid_label[i], preds[i]))
for i in range(len(valid_X)):
	print("%.2f  %.2f %.2f  %.2f  %.2f  %.2f  %.2f  ,and the class is %d " % (preds[i][0],preds[i][1],preds[i][2],preds[i][3],preds[i][4],preds[i][5],preds[i][6],np.argmax(valid_label[i])))

print(history.history.keys())
# summarize history for accuracy
# plt.plot(history.history['acc'])
# #plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# #plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.subplot(2, 1, 1)
plt.plot(history.history['acc'])
	#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
	#plt.show()
plt.savefig("augmented3Part1")