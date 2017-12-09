import time
stime = time.time()

import struct as st
import numpy as np

filename = {'images' : 'training_set/train-images.idx3-ubyte' ,'labels' : 'training_set/train-labels.idx1-ubyte'}

labels_array = np.array([])

data_types = {
        0x08: ('ubyte', 'B', 1),
        0x09: ('byte', 'b', 1),
        0x0B: ('>i2', 'h', 2),
        0x0C: ('>i4', 'i', 4),
        0x0D: ('>f4', 'f', 4),
        0x0E: ('>f8', 'd', 8)}

for name in filename.keys():
	if name == 'images':
		imagesfile = open(filename[name],'r+')
	if name == 'labels':
		labelsfile = open(filename[name],'r+')

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
nBytes = nImg*nR*nC
labelsfile.seek(8) #Since no. of items = no. of images and is already read
print "no. of images :: ",nImg
print "no. of rows :: ",nR
print "no. of columns :: ",nC

#Read all data bytes at once and then reshape
images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytes,imagesfile.read(nBytes))).reshape((nImg,nR,nC))
labels_array = np.asarray(st.unpack('>'+'B'*nImg,labelsfile.read(nImg))).reshape((nImg,1))

print labels_array
print labels_array.shape
print images_array.shape

print "Time of execution : %s seconds" % str(time.time()-stime)
