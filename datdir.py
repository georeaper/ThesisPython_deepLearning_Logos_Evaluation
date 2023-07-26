import gzip
import os
import sys
import timeit
import shutil

def creating_datasets(dir_home,dir_dest,train_per,test_per,val_per):
	onlyfiles = next(os.walk(dir_home))[2] #dir is your directory path as string
	length=len(onlyfiles)
	#print (length)
	#all images in one directory
	home_dire=dir_home

	#destination directory

	dest_dire=dir_dest
	

	set_train_dataset=int(length*train_per/100)
	set_test_dataset=int(length*test_per/100)
	set_val_dataset=int(length*val_per/100)


	final_length=set_train_dataset+set_test_dataset+set_val_dataset
	#print(final_length)
	#print(set_test_dataset)
	if(length>final_length):
		add_values=length-final_length

		set_test_dataset=set_test_dataset+add_values
	
	#final_length=set_train_dataset+set_test_dataset+set_val_dataset
	
	#print(final_length)
	filelist = os.listdir(home_dire)
	filelist.sort()
	count=0
	for i in filelist:
		if count >
		print (i)
		count=count+1
	
	

	

	return set_test_dataset,set_train_dataset,set_val_dataset




if __name__ == "__main__":

	x,y,z= creating_datasets("C:/Users/MasterPC/Desktop/photos/test/images/","C:/Users/MasterPC/Desktop/photos/test/images/",60,30,10)

	#print(x)

	
