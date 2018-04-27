#-*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2018-4-26"

# Demo to predict one image,plot results,add CRFlayer

import sys,os,pdb
import numpy as np
import cv2

import tensorflow as tf
from skimage.io  import imread,imsave
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from model import FCN32_test, FCN16_test, FCN8_test
from dataloader import Dataloader, Dataloader_small
from util import seg_gray_to_rgb,colormap

gray_to_rgb, rgb_to_gray = colormap()

# BGR mean pixel value
MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

CLASSES = ('__background__',
		   'aeroplane', 'bicycle', 'bird', 'boat',
		   'bottle', 'bus', 'car', 'cat', 'chair',
		   'cow', 'diningtable', 'dog', 'horse',
		   'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor')

config = {
'batch_num':1, 
'iter':100000, 
'num_classes':21, 
'max_size':(640,640),
'weight_decay': 0.0005,
'base_lr': 0.001,
'momentum': 0.9
}

def predict(path):
	'''
	use fcn to predict for segmentation
	'''
	model = FCN16_test(config)
	data_loader = Dataloader('val', config)

	saver = tf.train.Saver()
	ckpt = './models/FCN16_adam_iter_5000.ckpt'
	dump_path = './dataset/demo/'
	if not os.path.exists(dump_path):
		os.makedirs(dump_path)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		saver.restore(session, ckpt)
		print ('Model restored.')
		im = cv2.imread(path)
		im = cv2.resize(im,(640,640))
		im2 = np.expand_dims(im,0)
		feed_dict = {model.img: im2}
						
		pred = session.run(model.get_output('deconv'), feed_dict=feed_dict)
		#pdb.set_trace()
		
		annotated_label  = np.argmax(pred[0], axis=2)

		return annotated_label

def crf(root,original_image,annotated_label,output_image, use_2d = True):
	'''
	Original_image = Image which has to labelled
	Annotated image = Which has been labelled by some technique( FCN in this case)
	Output_image = The final output image after applying CRF
	Use_2d = boolean variable 
	if use_2d = True specialised 2D fucntions will be applied
	else Generic functions will be applied
	'''   
	
	# Converting annotated image to RGB if it is Gray scale
	colors, labels = np.unique(annotated_label, return_inverse=True)
	#Creating a mapping back to 32 bit colors
	colorize = np.empty((len(colors), 3), np.uint8)
	colorize[:,0] = (colors & 0x0000FF)
	colorize[:,1] = (colors & 0x00FF00) >> 8
	colorize[:,2] = (colors & 0xFF0000) >> 16  
	#Gives no of class labels in the annotated image
	n_labels = len(set(labels.flat)) 

	
	
	#Setting up the CRF model
	if use_2d :
		d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
		# get unary potentials (neg log probability)
		U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
		d.setUnaryEnergy(U)
		# This adds the color-independent term, features are the locations only.
		d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
						  normalization=dcrf.NORMALIZE_SYMMETRIC)
		# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
		d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
						   compat=10,
						   kernel=dcrf.DIAG_KERNEL,
						   normalization=dcrf.NORMALIZE_SYMMETRIC)
		
	#Run Inference for 5 steps 
	Q = d.inference(5)
	# Find out the most probable class for each pixel.
	MAP = np.argmax(Q, axis=0)#(409600, 3)
	# Convert the MAP (labels) back to the corresponding colors and save the image.
	# Note that there is no "unknown" here anymore, no matter what we had at first.
	MAP = colorize[MAP,:]
	seg_crf = MAP.reshape(original_image.shape)[:,:,0]
	seg_rgb = seg_gray_to_rgb(seg_crf, gray_to_rgb)

	cv2.imwrite(root+output_image,seg_rgb)

	return (MAP.reshape(original_image.shape))


if __name__ =="__main__":
	root = './dataset/demo/'
	path = root+'test.jpg'
	annotated_label = predict(path)
	original_im = imread(root+'test_origin.png')
	output = crf(root,original_im,annotated_label,'test_crf.png')