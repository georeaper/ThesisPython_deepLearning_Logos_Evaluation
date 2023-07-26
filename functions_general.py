import csv
from PIL import Image, ImageFilter
from resizeimage import resizeimage
import os
import numpy
import six.moves.cPickle as pickle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.utils import class_weight
def give_arguments(arg):
	csv_rows=[]
	spamReader = csv.reader(open('C:/Users/MasterPC/Desktop/running_keras_info.csv'), delimiter=';', quotechar='|')
	for row in spamReader:
		csv_rows.append(row)


	return csv_rows[arg]

def res_img_foler(diri,dir_dest):
	count=0
	countx=0
	county=0
	my_x=[]
	my_y=[]
	zeros=(1024,1024)
	filelist=os.listdir(diri)
	filelist.sort()
	ccount=20000
	for i in filelist:
		if(count<ccount):
			print(count)
			im= Image.open(diri+i)
			im=numpy.array(im)
			im=im[50:820,150:920]
			im = Image.fromarray(im)
			cover=resizeimage.resize_cover(im,[225,225])
			cover.save(dir_dest+i,im.format)
			count=count+1
			print(count)
			#imnp=numpy.array(im)
			#print(numpy.sum(im))
			
			# print(im.shape)
			# print(i)
			# if(imnp.shape==zeros):
			# 	#print("ok")
			# 	# print(im.shape)
			# 	# print(i)
			# 	im_new=imnp[50:820,150:920]
				
			# 	im = Image.fromarray(im_new)
			# 	cover=resizeimage.resize_cover(im,[225,225])
			# 	cover.save(dir_dest+i,im.format)
			# 	count=count+1


				#im.save(dir_dest+i)
				#im.show("your_file.png")
				#my_x.append(im_new)
				#y=find_label(i)
				#if(y==1):
					#countx=countx+1
				#else:
					#county=county+1
				#print(y)
				#my_y.append(y)
				#count=count+1
			#else:
				#print(im.shape)
		else:
			break
def gray_one_to_seven(pic):
	new_img=[]
	#im= Image.open(pic)
	im=pic
	im=numpy.array(im)
	#print("shape1 before")
	#print(im.shape)
	im2=numpy.reshape(im,50625)
	#print("shape2 before")
	#print(im2.shape)
	for i in im2:
		d=7*i/255
		d=round(d,0)
		new_img.append(d)
	im2=numpy.array(new_img)
	#print("im2")
	#print(im2)
	im=numpy.reshape(im2,(225,-1))
	#print("im")
	#print(im)
	#print(im.shape)

	return im

	# im=im[50:820,150:920]
	# im = Image.fromarray(im)
	# cover=resizeimage.resize_cover(im,[225,225])
	# cover.save(dir_dest+i,im.format)
def to_gray(diri,dir_dest):
	filelist=os.listdir(diri)
	filelist.sort()
	count=0
	for i in filelist:
		print(count)
		im= Image.open(diri+i)
		#print(im.mode)
		#im = im.convert('F')
		#new_img=gray_one_to_seven(im)

		#im = Image.fromarray(new_img,"I")
		im_sharp = im.filter( ImageFilter.EDGE_ENHANCE )
		#Saving the filtered image to a new file
		im_sharp.save( dir_dest+i)
		#im.show()
		#print("echo")
		#im.save(dir_dest+i)
		count=count+1

def logos_pkl():
	images=open('C:/Users/MasterPC/pythonprogs/images_logos21.pickle', 'rb')
	images=pickle.load(images)

	data=open('C:/Users/MasterPC/pythonprogs/y_logos.pickle', 'rb')

	y_logos=pickle.load(data)

	indices,active,comple_x,depth=y_logos
	#print(active)


	train_images=images[0:208]
	train_images=numpy.array(train_images)

	test_images=images[206:208]
	test_images=numpy.array(test_images)

	Cact=[]
	for i in range(0,208):
		mean_agg=int(active[i][0])+int(active[i][1])+int(active[i][2])
		mean_c=round(mean_agg/3)
		#print(int(mean_c))

		#--------
		#below we are appending the value of the first rate and we ignoring the rest
		Cact.append(int(mean_c))
	
	train_act_y=Cact[0:208]
	test_act_y=Cact[206:209]

	#print(test_act_y)
	train_act_y=numpy.array(train_act_y)
	test_act_y=numpy.array(test_act_y)
	x=Counter(train_act_y)
	print(x)
	
	class_weight1 = class_weight.compute_class_weight('balanced',
                                                 numpy.unique(train_act_y),
                                                 train_act_y)
	print("Class weights :",class_weight1)
	#print("ooooooo")
	print(train_act_y)
	#print("ooooooo")
	train_Y_one_hot=to_categorical(train_act_y)
	test_Y_one_hot=to_categorical(test_act_y)
	# print(train_Y_one_hot)
	classes=numpy.unique(train_act_y)
	nClasses=len(classes)
	print("Total number of outputs : ", nClasses)
	print('Output classes : ',classes)
	print(train_images.shape)

	train_X=train_images.reshape(-1,225,225,3)

	#test_X=x_test.reshape(-1,225,225,1)

	

	#print(test_X.shape)

	train_X,valid_X,train_label,valid_label = train_test_split(train_images, train_Y_one_hot, test_size=0.15, random_state=20)
	# print(train_X.shape)
	print(train_label.shape)


	return train_X,valid_X,train_label,valid_label,class_weight1

def lung_dt():
	train_file=open('C:/Users/MasterPC/pythonprogs/train225x225_10000_edge.pickle', 'rb')
	data=pickle.load(train_file)
	x,y=data
	x_train=numpy.array(x)
	y_train=numpy.array(y)
	train_file.close()
	print(x_train[1].shape)
	#print("Train file Loaded....Loading Test file")
	test_file=open('C:/Users/MasterPC/pythonprogs/test225x225_200_edge.pickle', 'rb')
	data=pickle.load(test_file)
	x,y=data
	x_test=numpy.array(x)
	y_test=numpy.array(y)
	test_file.close()

	print("training Data shapes : ",x_train.shape,y_train.shape)
	print("test Data shapes : ",x_test.shape,y_test.shape)

	classes=numpy.unique(y_train)
	nClasses=len(classes)
	print("Total number of outputs : ", nClasses)
	print('Output classes : ',classes)

	train_X=x_train.reshape(-1,225,225,3)

	#test_X=x_test.reshape(-1,225,225,3)

	print(train_X.shape)

	#print(test_X.shape)

	train_X=train_X.astype('float32')
	#test_X=test_X.astype('float32')
	train_X=train_X/255
	#test_X=test_X/255

	train_Y_one_hot=to_categorical(y_train)
	#test_Y_one_hot=to_categorical(y_test)

	print('Original Label: ',y_train[0])
	print('After conversion: ',train_Y_one_hot[0])

	train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.1, random_state=13)

	return train_X,valid_X,train_label,valid_label


	# def logos_pkl_create_more():

		#in this we are going to try to create more classes by rotating the images

#####test me augmented logos#####
def train_valid_augmented_data(diri):
	data=open(diri, 'rb')

	logo_data=pickle.load(data)	
	x,y=logo_data
	x=numpy.array(x)
	train_act_y=numpy.array(y)


	xcount=Counter(train_act_y)
	print(xcount)

	#test_act_y=Cact[206:209]
	#print(test_act_y)
	train_act_y=numpy.array(train_act_y)
	#test_act_y=numpy.array(test_act_y)
	class_weight1 = class_weight.compute_class_weight('balanced', numpy.unique(train_act_y),train_act_y)
	print(class_weight1)
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
	print("shape of train_act_y",train_act_y[1])
	train_Y_one_hot=to_categorical(train_act_y)
	#train_Y_one_hot=train_act_y
	#print(train_Y_one_hot)
	classes=numpy.unique(train_act_y)
	nClasses=len(classes)
	print("Total number of outputs : ", nClasses)
	print('Output classes : ',classes)
	#print(x.shape)

	train_X=x.reshape(-1,225,225,3)

	#test_X=x_test.reshape(-1,225,225,1)

	#print(train_X.shape)

	#print(test_X.shape)

	train_X,valid_X,train_label,valid_label = train_test_split(x, train_Y_one_hot, test_size=0.1, random_state=6)

	return train_X,valid_X,train_label,valid_label,class_weight1