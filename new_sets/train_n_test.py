import numpy
import csv
#import gzip
import os
#import sys
#import timeit
#import shutil
from PIL import Image
import six.moves.cPickle as pickle

def lung_dataset(dir_train):
	my_x=[]
	my_y=[]
	filelist=os.listdir(dir_train)
	filelist.sort()
	count=0
	for i in filelist:
		
		print(count)
		im= Image.open(dir_train+i)
		im=numpy.array(im)

		my_x.append(im)
		y=find_label(i)
		#print(y)
		my_y.append(y)
		count=count+1

	c=my_x,my_y
	return c


def find_label(name):
	spamReader = csv.reader(open('C:/Users/MasterPC/pythonprogs/LUNGS/readme/DE.csv', newline=''), delimiter=',', quotechar='|')
	for row in spamReader:
		if(row[0]==name):
			if (row[1]=="No Finding"):
				x=1
			else:
				x=0
			break

	return x


if __name__ == "__main__":
	c= lung_dataset("C:/Users/MasterPC/pythonprogs/LUNGS/train_images/")
	with open('train.pickle', 'wb') as f:
		pickle.dump(c, f, pickle.HIGHEST_PROTOCOL)

	f.close()