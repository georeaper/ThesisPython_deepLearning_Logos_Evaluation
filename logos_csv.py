import numpy
import csv
#import gzip
import os
#import sys
#import timeit
#import shutil
from PIL import Image
import six.moves.cPickle as pickle

from resizeimage import resizeimage
indices=[]
activeness=[]
complexity=[]
depth=[]
print("start")
spamReader = csv.reader(open('C:/Users/MasterPC/Desktop/DeepComputionalAesthetics/elaborateness.csv', newline=''), delimiter=';', quotechar='|')
for row in spamReader:
		#test="image_new_dim"+row[0]
		#print(row[0])
		indices.append(row[0:7])
		activeness.append(row[8:11])
		# activeness.append(row[8:11])
		# activeness.append(row[8:11])
		# activeness.append(row[8:11])
		complexity.append(row[11:14])
		depth.append(row[14:17])

#print("end")

test= indices,activeness,complexity,depth
#print(depth)
#-------------------------
#-------------------------
#-------------------------
my_x=[]
dir_train="C:/Users/MasterPC/Desktop/DeepComputionalAesthetics/WIPO/"
filelist=os.listdir(dir_train)

count=0
countx=0
county=0

for i in filelist:
		##################
		#print(count)
	#print(i)
	im= Image.open(dir_train+i)
	# image.thumbnail(resample_size)
	# image = image.convert("RGB")
	# im = im[:, :, :3]
	#im = im.convert('RGB')
	im = resizeimage.resize_contain(im, [225, 225])
	im=numpy.array(im)
	im=im[: , : , :3]
	print(im.shape)
	my_x.append(im)


#print(my_x)

# f=open('images_logos21.pickle', 'wb')
# pickle.dump(my_x, f, pickle.HIGHEST_PROTOCOL)
# f.close()

# f=open('y_logos.pickle', 'wb')
# pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
# f.close()