import six.moves.cPickle as pickle
import numpy


pkl_file = open('C:/Users/MasterPC/pythonprogs/test.pickle', 'rb')


data1 = pickle.load(pkl_file)

c,p= data1

c=numpy.array(c)
p=numpy.array(p)
#print(c)
print(p.shape)
#print(zeros)
# if (c.shape==zeros):
# 	print("Horrey")
# else:
# 	print("something went wrong")