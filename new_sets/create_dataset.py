from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from PIL import Image
def mamms_dataset():
    print ('creating dataset')
    batch_size=2
    ##datasets = load_data(dataset)
    my_x=[]
    my_y=[]
    with open("C:/Users/MasterPC/pythonprogs/test_code/info.txt") as f:
        i=0
    #test=[][]
        for line in f:
            #i+=1
            strlist=line.split(" ")
            if(len(strlist)>2):
                if(strlist[2]!='NORM'):
                    #print strlist
                    if(strlist[3]=='M'):
                        src='C:/Users/MasterPC/pythonprogs/test_code/all-mias/'+strlist[0]+'.pgm'
                        im = Image.open(src)
                        im=numpy.array(im, dtype='float64')
                        # //im = numpy.reshape(im, 1048576)
                        my_x.append(im)
                        my_y.append(33)
                        #dest2='C:/Users/Giorgos/Documents/pyprogs/m/'+strlist[0]+'.pgm'
                        #print src

                    #shutil.copy(src,dest2)
                    else:
                        src='C:/Users/MasterPC/pythonprogs/test_code/all-mias/'+strlist[0]+'.pgm'
                        im = Image.open(src)
                        im=numpy.array(im, dtype='float64')
                        # im = numpy.reshape(im, 1048576)
                        my_x.append(im)
                        my_y.append(2)
                        #dest2='C:/Users/Giorgos/Documents/pyprogs/b/'+strlist[0]+'.pgm'
                        #shutil.copy(src,dest2)
                else:
                    src='C:/Users/MasterPC/pythonprogs/test_code/all-mias/'+strlist[0]+'.pgm'
                    im = Image.open(src)
                    im=numpy.array(im, dtype='float64')
                    # im = numpy.reshape(im, 1048576)
                    my_x.append(im)
                    my_y.append(1)
                    #dest2='C:/Users/Giorgos/Documents/pyprogs/n/'+strlist[0]+'.pgm'
                    #shutil.copy(src,dest2)


    train_x=my_x[0:200]
    print(len(train_x[0]))
    
    train_y=my_y[0:200]
    test_x=my_x[201:290]
    test_y=my_y[201:290]
    val_x=my_x[291:323]
    val_y=my_y[291:323]
    # print (my_y[0])
    # print (val_x)
    # print (val_y)
    train_set_x = theano.shared(numpy.asarray(train_x, dtype='float64'),borrow=True)
    train_set_y = theano.shared(numpy.asarray(train_y, dtype='float64'),borrow=True)
    test_set_x = theano.shared(numpy.asarray(test_x, dtype='float64'),borrow=True)
    test_set_y = theano.shared(numpy.asarray(test_y, dtype='float64'),borrow=True)
    valid_set_x = theano.shared(numpy.asarray(val_x, dtype='float64'),borrow=True)
    valid_set_y = theano.shared(numpy.asarray(val_y, dtype='float64'),borrow=True)