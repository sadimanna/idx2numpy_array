import time
stime = time.time()

import struct as st
import numpy as np

filename = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
images_array = np.array([])
labels_array = np.array([])

for name in filename.keys():
	if name == 'images':
		imagesfile = open(filename[name],'r+')
	if name == 'labels':
		labelsfile = open(filename[name],'r+')

#offset = 0004 for number of images
#offset = 0008 for number of rows
#offset = 0012 for number of columns
#32-bit integer (32 bits = 4 bytes)
imagesfile.seek(4)
nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images/labels
nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
nC = st.unpack('>I',imagesfile.read(4))[0] #num of columns
labelsfile.seek(8) #Since no. of items = no. of images and is already read
print "no. of images :: ",nImg
print "no. of rows :: ",nR
print "no. of columns :: ",nC

temp_array = np.array([])
images10000_array = np.array([])
for i in xrange(1,nImg+1):
	#Read labels
	labels_array = np.append(labels_array,st.unpack('>B',labelsfile.read(1))[0])
	#Read training images
	if temp_array.size == 0:
		#invert the image as 255 is white and 0 is black
		temp_array = 255 - np.asarray(st.unpack('>784B',imagesfile.read(784))).reshape((nR,nC))
	else:
		nextimage = 255 - np.asarray(st.unpack('>784B',imagesfile.read(784))).reshape((nR,nC))
		if len(temp_array.shape)==2:
			temp_array = np.vstack((temp_array[None],nextimage[None]))
		else:
			temp_array = np.vstack((temp_array,nextimage[None]))
	
	#Extra stuffs to speed up the stacking process (took 51.804361105 seconds in my case)
	#Stacking each 1000 block to form a block of 10000
	if i%1000==0 and i != 0:
		if images10000_array.size == 0:
			images10000_array = temp_array
		else:
			images10000_array = np.vstack((images10000_array,temp_array))
		temp_array = np.array([])
		print "Time taken :: ",time.time()-stime
	#Stacking each 10000 block to form the whole dataset
	if i%10000==0 and i != 0: 
		if images_array.size == 0:
			images_array = images10000_array
		else:
			images_array = np.vstack((images_array,images10000_array))
		images10000_array = np.array([])
		print (float(i)/nImg)*100,"% complete..."

print labels_array.shape
print images_array.shape

print "Time of execution : %s seconds" % str(time.time()-stime)
