# Training code


import numpy as np
import tensorflow as tf
from model import FCN32, FCN16, FCN8
from dataloader import Dataloader
import pdb


config = {
'batch_num':1, 
'iter':10000, 
'num_classes':21, 
'max_size':(640,640),
'weight_decay': 0.0005,
'base_lr': 0.0001,
'momentum': 0.9
}

if __name__ == '__main__':
	#pdb.set_trace()
	#load data
	data_loader = Dataloader('train', config)
	minibatch  = data_loader.get_next_minibatch()

	# Load pre-trained model
	model_path = './models/VGG_imagenet.npy'
	data_dict = np.load(model_path,encoding='latin1').item()
	# Set up model 
	model = FCN16(config)
	loss_list = []
	DECAY = False    # decay flag
	init = tf.initialize_all_variables()

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
		session.run(init)
		model.load(data_dict, session)
		saver = tf.train.Saver()

		loss = 0
		for i in (range(config['iter'])):
			minibatch = data_loader.get_next_minibatch()
			feed_dict = {model.img: minibatch[0],
						model.seg: minibatch[1],
						model.mask: minibatch[2]}
			_, temp_loss = session.run([model.train_op, model.loss], feed_dict=feed_dict)
			loss += temp_loss

			loss_list.append(temp_loss)
			print('Iter:'+str(i) + '   Loss: ' + str(temp_loss))

			# Learning rate decay
			if len(loss_list) > 100 and not DECAY:
				avg = sum(loss_list[-100::]) / 100.0
				if avg <= 0.4:
					model.base_lr /= 10
					DECAY = True
			# Monitor
			if i % 20 == 0 and i != 0:
				loss /= 20
				print ('Iter: {}'.format(i) + '/{}'.format(config['iter']) + ', Mean Loss = ' + str(loss))
				loss = 0
			# Write to saver
			if i % 1000 == 0 and i != 0:
				saver.save(session, './models/FCN16_iter_'+str(i)+'.ckpt')

