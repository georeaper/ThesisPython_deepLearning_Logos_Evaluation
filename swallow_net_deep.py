import six.moves.cPickle as pickle
import numpy
from keras.utils import to_categorical
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from functions_general import train_valid_augmented_data
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
from sklearn.utils import class_weight
#from keras_functions import EarlyStoppingByLossVal as early
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback
#print(sys.argv[0])
#print(sys.argv[1])
#print(sys.argv[2])
temp='model'+sys.argv[1]+'.png'
print(temp)
n1=32
n2=64
n3=128
d1=256

train_X, valid_X, train_label, valid_label, class_weight1=train_valid_augmented_data('augmented_class1_logos_complexity.pickle')

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
from sklearn.metrics import mean_absolute_error

class EarlyStoppingByAcc(Callback):
    def __init__(self, monitor='acc', value=0.90, verbose=0):
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


fashion_model = Sequential()

fashion_model.add(Conv2D(n1, kernel_size=(2,2),activation='relu',input_shape=(225,225,3),padding='same'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))


fashion_model.add(Conv2D(n2, (2,2), activation='relu',padding='same'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))


fashion_model.add(Conv2D(n3, (2,2), activation='relu',padding='same'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))

fashion_model.add(Flatten())
fashion_model.add(Dense(d1, activation='relu'))
fashion_model.add(Dense(d1, activation='relu'))

fashion_model.add(Dense(7, activation='softmax'))


sgd = optimizers.SGD(lr=0.01)
fashion_model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
fashion_model.summary()
from keras.utils import plot_model
plt_test=plot_model(fashion_model, to_file='model_layers'+sys.argv[1]+'.png')

with tf.device('/gpu:0'):
	#history = fashion_model.fit(train_X, train_label, batch_size=20,epochs=20,verbose=1,validation_data=(valid_X, valid_label))
	history = fashion_model.fit(train_X, train_label,class_weight=class_weight1, batch_size=20,epochs=20,verbose=2,callbacks=[EarlyStoppingByAcc()])

ynew = fashion_model.predict_classes(valid_X)
count=0
y_true=[]
y_pred=[]
for i in range(len(valid_X)):

	print("X=%s, Predicted=%s" % (valid_label[i], ynew[i]))
	validl=numpy.argmax(valid_label[i], axis=None, out=None)
	#print(validl)
	if (validl==ynew[i]):
		count=count+1
	y_true.append(validl)
	y_pred.append(ynew[i])




print("percent_acc",count/len(valid_X))

cm_valid = confusion_matrix(y_true, ynew, labels=[0, 1, 2, 3, 4, 5, 6])
print("confusion_matrix upcoming for validation")
print(cm_valid)

mae=mean_absolute_error(y_true, y_pred)
print("MAE is :",mae)
switcher=2
print(history.history.keys())

if (switcher==1):

	
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
else:
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
	plt.savefig(temp)