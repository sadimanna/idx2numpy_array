import time
import numpy as np
from scipy.misc import imsave

stime = time.time()
from idx2ndarray import train_images_array,test_images_array,train_labels_array,test_labels_array
print "\nTime for loading numpy arrays from idx2ndarray :: "+str(time.time()-stime)+" seconds\n"

trainImgshape = train_images_array.shape
trainLabelshape = train_labels_array.shape
testImgshape = test_images_array.shape
testLabelshape = test_labels_array.shape

stime = time.time()
training_folderName = 'training_set_images/'
fileNameLen = len(str(trainImgshape[0]))
nIter = trainImgshape[0]+1
for n in xrange(1,nIter):
	filename = '0'*(fileNameLen - len(str(n)))+str(n)+'.jpg'
	#print filename
	imsave(training_folderName+filename,train_images_array[n-1,:,:])
print "Time for converting training dataset array to images :: "+str(time.time()-stime)+" seconds\n"

stime = time.time()
test_folderName = 'test_set_images/'
fileNameLen = len(str(testImgshape[0]))
nIter = testImgshape[0]+1
for n in xrange(1,nIter):
	filename = '0'*(fileNameLen - len(str(n)))+str(n)+'.jpg'
	#print filename
	imsave(test_folderName+filename,test_images_array[n-1,:,:])
print "Time for converting test dataset array to images :: "+str(time.time()-stime)+" seconds\n"

stime = time.time()
trainingLabelFileName = 'training_set_labels'
np.save(trainingLabelFileName,train_labels_array,allow_pickle=False,fix_imports=False)
print "Time for saving Training Labels array as .npy file :: "+str(time.time()-stime)+" seconds\n"

stime = time.time()
testLabelFileName = 'test_set_labels'
np.save(testLabelFileName,test_labels_array,allow_pickle=False,fix_imports=False)
print "Time for saving Test Labels array as .npy file :: "+str(time.time()-stime)+" seconds\n"

#saved as .npy
