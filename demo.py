#-*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2018-4-26"

# Demo to predict one image,plot results

import numpy as np
import tensorflow as tf
from model import FCN32_test, FCN16_test, FCN8_test
from dataloader import Dataloader, Dataloader_small
from util import get_original_size, seg_gray_to_rgb
import matplotlib.pyplot as plt
import cv2
import pdb
import os


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

if __name__ =="__main__":
	root = './dataset/demo/'
	path = root+'test.jpg'
	annotated_label = predict(path)
	seg_rgb = seg_gray_to_rgb(annotated_label, data_loader.gray_to_rgb)
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
	ax1.imshow(seg_rgb)
	ax2.imshow(im)
	plt.show()
	cv2.imwrite(dump_path+path.split('/')[-1].split('.')[0]+'_seg.png', seg_rgb)
	cv2.imwrite(dump_path+path.split('/')[-1].split('.')[0]+'_origin.png', im)
