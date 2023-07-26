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
	zeros=(225,225,3)
	for i in filelist:
		##################
		#print(count)
		if(count<ccount):
			im= Image.open(dir_train+i)
			im=numpy.array(im)
			im=to_rgb1(im)
			im = numpy.resize(im, (225, 225, 3))
			print(im.shape)
			# print(i)
			if(im.shape==zeros):
				print(count)
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
	c=my_x,my_y
	return c
		
def lung_dataset_reshape(dir_train,ccount):
	my_x=[]
	my_y=[]
	filelist=os.listdir(dir_train)
	filelist.sort()
	count=0
	countx=0
	county=0
	zeros=(1024,1024)
	for i in filelist:
		##################
		#print(count)
		if(count<ccount):
			im= Image.open(dir_train+i)
			im=numpy.array(im)
			#print(numpy.sum(im))
			#print(numpy.sum(im_new))
			# print(im.shape)
			# print(i)
			if(im.shape==zeros):
				print(count)
				#print("ok")
				# print(im.shape)
				#print(i)
				im_new=im[50:820,150:920]
				
				#im = Image.fromarray(im_new)
				#im.save(dir_dest+i)
				#im.show("your_file.png")
				my_x.append(im_new)
				y=find_label(i)
				if(y==1):
					countx=countx+1
				else:
					county=county+1
				#print(y)
				my_y.append(y)
				count=count+1
				
			#else:
				#print(im.shape)
		else:
			break
	
		


		
		# if (count==400):
		# 	print("50%")

	c=my_x,my_y
	return c,countx,county


def find_label(name):
	x=0
	#print(name)
	
	spamReader = csv.reader(open('C:/Users/MasterPC/pythonprogs/LUNGS/readme/DE.csv', newline=''), delimiter=',', quotechar='|')
	for row in spamReader:
		#test="image_new_dim"+row[0]
		test=row[0]
		if(test==name):
			if (row[1]=="No Finding"):
				x=1
				
			else:
				x=0
				
			break

	return x

def to_rgb1(im):
    # I think this will be slow
    #print(im.shape)
	if im.shape==(225,225,1):
	    print(im.shape)
	    w, h = im.shape

	    ret = numpy.empty((w, h, 3), dtype=numpy.uint8)
	    ret[:, :, 0] = im
	    ret[:, :, 1] = im
	    ret[:, :, 2] = im
	else:
		ret=im
	print(ret.shape)
	return ret

if __name__ == "__main__":
	#print("In galaxy far far away,the rebels start to  re-create the lung_dataset..The first method...Training")
	c= lung_dataset("D:/pythonprogs/Lungs/test/edge_images/",6000)
	# iii,jjj=c
	# print(iii.shape)
	f=open('train225x225_6000_edge2.pickle', 'wb')
	pickle.dump(c, f, pickle.HIGHEST_PROTOCOL)
	f.close()
	print("phase 2")
	#f.open('train225x225_6000.pickle', 'wb')
	c= lung_dataset("D:/pythonprogs/Lungs/test/test_edge_images/",200)
	# iii,jjj=c
	# print(iii.shape)
	f=open('test225x225_200_edge.pickle', 'wb')
	pickle.dump(c, f, pickle.HIGHEST_PROTOCOL)


	f.close()

	#print(getx,gety)
	# f.close()

	