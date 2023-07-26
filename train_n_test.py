import numpy
import csv
#import gzip
import os
#import sys
#import timeit
#import shutil
from PIL import Image
import six.moves.cPickle as pickle


def lung_dataset(dir_train,ccount):
	my_x=[]
	my_y=[]
	filelist=os.listdir(dir_train)
	filelist.sort()
	count=0
	zeros=(1024,1024)
	for i in filelist:
		##################
		#print(count)
		if(count<ccount):
			im= Image.open(dir_train+i)
			im=numpy.array(im)
			# print(im.shape)
			# print(i)
			if(im.shape==zeros):
				#print("ok")
				# print(im.shape)
				# print(i)
				my_x.append(im)
				y=find_label(i)
				#print(y)
				my_y.append(y)
				count=count+1
			else:
				print(im.shape)
		else:
			break
		

		


		
		# if (count==400):
		# 	print("50%")

	c=my_x,my_y
	return c


def find_label(name):
	spamReader = csv.reader(open('C:/Users/MasterPC/pythonprogs/LUNGS/readme/DE.csv', newline=''), delimiter=',', quotechar='|')
	for row in spamReader:
		#print(row[0])
		if(row[0]==name):
			if (row[1]=="No Finding"):
				x=1
			else:
				x=0
			break

	return x


if __name__ == "__main__":
	#print("In galaxy far far away,the rebels start to  re-create the lung_dataset..The first method...Training")
	c= lung_dataset("D:/pythonprogs/Lungs/test/train/",1000)
	f=open('train1000.pickle', 'wb')
	pickle.dump(c, f, pickle.HIGHEST_PROTOCOL)

	f.close()
	print("While the first phase has been completed,the rebels will start the testing method..Method 2..The TEST")
	c= lung_dataset("D:/pythonprogs/Lungs/test/test/",100)
	f= open('test12.pickle', 'wb')
	pickle.dump(c, f, pickle.HIGHEST_PROTOCOL)

	f.close()

	#print("The rebels was able to complete the training and the testing and will start to make a massive attack against the republic.")