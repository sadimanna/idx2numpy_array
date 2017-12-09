import time
stime = time.time()

import struct as st
import numpy as np

trainingfilenames = {'images' : 'training_set/train-images.idx3-ubyte' ,'labels' : 'training_set/train-labels.idx1-ubyte'}
testfilenames = {'images' : 'test_set/t10k-images.idx3-ubyte' ,'labels' : 'test_set/t10k-labels.idx1-ubyte'}

data_types = {
        0x08: ('ubyte', 'B', 1),
        0x09: ('byte', 'b', 1),
        0x0B: ('>i2', 'h', 2),
        0x0C: ('>i4', 'i', 4),
        0x0D: ('>f4', 'f', 4),
        0x0E: ('>f8', 'd', 8)}

#For training dataset
for name in trainingfilenames.keys():
	if name == 'images':
		imagesfile = open(trainingfilenames[name],'r+')
	if name == 'labels':
		labelsfile = open(trainingfilenames[name],'r+')

imagesfile.seek(0)
magic = st.unpack('>4B',imagesfile.read(4))
if(magic[0] and magic[1])or(magic[2] not in data_types):
	raise ValueError("File Format not correct")

nDim = magic[3]
print "Data is ",nDim,"-D"

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

nBatch = 10000
nIter = nImg/nBatch+1
nBytes = nBatch*nR*nC
nBytesTot = nImg*nR*nC
#Read all data bytes at once and then reshape
#Reading the labels
labels_array = np.asarray(st.unpack('>'+'B'*nImg,labelsfile.read(nImg))).reshape((nImg,1))
images_array = np.array([])
for i in xrange(1,nIter):
	temp_images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytes,imagesfile.read(nBytes))).reshape((nBatch,nR,nC))
	#Extra stuffs to speed up the stacking process
	#Stacking each nBatch block to form a larger block
	if images_array.size == 0:
		images_array = temp_images_array
	else:
		images_array = np.vstack((images_array,temp_images_array))
	temp_images_array = np.array([])
	print "Time taken :: ",time.time()-stime
	print (float(i)/nIter)*100,"% complete..."


print labels_array.shape
print images_array.shape

print "Time of execution : %s seconds" % str(time.time()-stime)
