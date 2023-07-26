import os
import shutil
import gzip
import sys
import timeit


dir_home="D:/pythonprogs/Lungs/test/images/"
dir_test="D:/pythonprogs/Lungs/test/test/"
dir_train="D:/pythonprogs/Lungs/test/train/"
onlyfiles = next(os.walk(dir_home))[2]

y=[200,800] #orismos sta posa tha ginontai oi allages
x=[0,1] #test kai training images

k=799 #arxiko
count=0
stage=1
filelist = os.listdir(dir_home)
filelist.sort()
t=0
ccount=0
limit_data=4000
for i in filelist:
	if(ccount<limit_data):
		#print("start")
		if (t<k) :
			if (stage==0):
				dir_dest=dir_test
			else:
				dir_dest=dir_train
			shutil.copyfile(dir_home+i,dir_dest+i)
			#shutil.move
			#print(stage + " stage ")
			count=count+1
			t=t+1 #ginetai i allagi stin else otan ftasi to orio
		

		else:
			#print(count)
			#print("++++++++++++++++++++++++++++++++++++++++++++")
			if(stage==1):
				stage=0
				dir_dest=dir_test
			else:
				stage=1
				dir_dest=dir_train
			count=0
			k=k+y[stage]
			count=count+1 #den kserw ton logo poy uparxei
			shutil.copyfile(dir_home+i,dir_dest+i)
			t=t+1 #ginontai oi allages se if kai else
	else:
		print("break")

		break
	ccount=ccount+1