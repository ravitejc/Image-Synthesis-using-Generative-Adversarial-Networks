import model
import argparse
import pickle
import scipy.misc
import random
import os
import shutil

import tensorflow as tf
import numpy as np

from os.path import join
from Utils import image_processing


def load_trained_data(data_directory, dataset, caption_vector_length, n_classes) :
	if dataset == 'flowers' :
		string_captions_flower = pickle.load(
			open(join(data_directory, 'flowers', 'flowers_caps.pkl'), "rb"))

		image_classes = pickle.load(
			open(join(data_directory, 'flowers', 'flower_tc.pkl'), "rb"))

		enc_captions_flower = pickle.load(
			open(join(data_directory, 'flowers', 'flower_tv.pkl'), "rb"))
		train_img_ids = pickle.load(
			open(join(data_directory, 'flowers', 'train_ids.pkl'), "rb"))
		validate_img_ids = pickle.load(
			open(join(data_directory, 'flowers', 'val_ids.pkl'), "rb"))

		maximum_captions_len = caption_vector_length
		train_n_imgs = len(train_img_ids)
		validate_n_imgs = len(validate_img_ids)

		return {
			'image_list'    : train_img_ids,
			'captions'      : enc_captions_flower,
			'data_length'   : train_n_imgs,
			'classes'       : image_classes,
			'n_classes'     : n_classes,
			'maximum_captions_len'  : maximum_captions_len,
			'val_img_list'  : validate_img_ids,
			'val_captions'  : enc_captions_flower,
			'val_data_len'  : validate_n_imgs,
			'str_captions'  : string_captions_flower
		}

	else :
		raise Exception('Failure: Dataset is not Found')


def set_directory_paths(arguments):
	model_directory = join(arguments.data_dir, 'training', arguments.model_name)
	if not os.path.exists(model_directory):
		os.makedirs(model_directory)

	model_chkpnts_directory = join(model_directory, 'checkpoints')
	if not os.path.exists(model_chkpnts_directory):
		os.makedirs(model_chkpnts_directory)

	model_summaries_directory = join(model_directory, 'summaries')
	if not os.path.exists(model_summaries_directory):
		os.makedirs(model_summaries_directory)

	model_samples_directory = join(model_directory, 'samples')
	if not os.path.exists(model_samples_directory):
		os.makedirs(model_samples_directory)

	model_val_samples_directory = join(model_directory, 'val_samples')
	if not os.path.exists(model_val_samples_directory):
		os.makedirs(model_val_samples_directory)

	return model_directory, model_chkpnts_directory, model_samples_directory, \
		   model_val_samples_directory, model_summaries_directory


def save_viz_val(data_directory, images_generated, image_files, image_captions_tensor,
				 image_id, image_size, id):

	images_generated = np.squeeze(np.array(images_generated))
	for i in range(0, images_generated.shape[0]) :
		img_directory = join(data_directory, str(image_id[i]))
		if not os.path.exists(img_directory):
			os.makedirs(img_directory)

		real_img_path = join(img_directory,
							   '{}.jpg'.format(image_id[i]))
		if os.path.exists(img_directory):
			real_imgs_255 = image_processing.load_image_array(image_files[i],
															  image_size, image_id[i], mode='val')
			scipy.misc.imsave(real_img_path, real_imgs_255)

		caps_directory = join(img_directory, "caps.txt")
		if not os.path.exists(caps_directory):
			with open(caps_directory, "w") as text_file:
				text_file.write(image_captions_tensor[i] + "\n")

		fake_imgs_255 = images_generated[i]
		scipy.misc.imsave(join(img_directory, 'fake_image_{}.jpg'.format(id)),
		                  fake_imgs_255)


def save_vis(data_directory, correct_images, images_generated, image_files,
			 image_captions, image_ids) :

	shutil.rmtree(data_directory)
	os.makedirs(data_directory)

	for i in range(0, correct_images.shape[0]) :
		correct_images_255 = (correct_images[i, :, :, :])
		scipy.misc.imsave(join(data_directory,
			   '{}_{}.jpg'.format(i, image_files[i].split('/')[-1])),
						  correct_images_255)

		fake_imgs_255 = (images_generated[i, :, :, :])
		scipy.misc.imsave(join(data_directory, 'fake_image_{}.jpg'.format(
			i)), fake_imgs_255)

	string_captions = '\n'.join(image_captions)
	str_img_ids = '\n'.join([str(img_id) for img_id in image_ids])
	with open(join(data_directory, "caps.txt"), "w") as text_f:
		text_f.write(string_captions)
	with open(join(data_directory, "ids.txt"), "w") as text_f:
		text_f.write(str_img_ids)


def get_captions_validation(batch_size, data, dataset, data_directory):

	if dataset == 'flowers':
		data_captions = np.zeros((batch_size, data['max_caps_len']))

		batch_ids = np.random.randint(0, data['val_data_len'],
									  size = batch_size)
		img_ids = np.take(data['val_img_list'], batch_ids)
		img_files = []
		img_caps = []
		for idx, img_id in enumerate(img_ids) :
			img_file = join(data_directory,
			                  'flowers/jpg/' + img_id)
			random_caption = random.randint(0, 4)
			data_captions[idx, :] = \
				data['val_captions'][img_id][random_caption][
				0:data['max_caps_len']]

			img_caps.append(data['str_captions']
			                  [img_id][random_caption])
			img_files.append(img_file)

		return data_captions, img_files, img_caps, img_ids
	else:
		raise Exception('Failure: No Dataset found')


def get_trained_batch(batch_number, batch_size, img_size, noise_dim, split,
					  data_directory, dataset, data = None) :
	if dataset == 'flowers':
		correct_imgs = np.zeros((batch_size, img_size, img_size, 3))
		wrong_imgs = np.zeros((batch_size, img_size, img_size, 3))
		captions = np.zeros((batch_size, data['max_caps_len']))
		correct_classes = np.zeros((batch_size, data['n_classes']))
		wrong_classes = np.zeros((batch_size, data['n_classes']))

		count = 0
		img_files = []
		img_caps = []
		img_ids = []
		for i in range(batch_number * batch_size,
					   batch_number * batch_size + batch_size) :
			idx = i % len(data['image_list'])
			img_file = join(data_directory,
			                  'flowers/jpg/' + data['image_list'][idx])

			img_ids.append(data['image_list'][idx])

			img_array = image_processing.load_image_array_flowers(img_file,
																  img_size)
			correct_imgs[count, :, :, :] = img_array

			# Improve this selection of wrong image
			wrong_img_id = random.randint(0,
										  len(data['image_list']) - 1)
			wrong_img_file = join(data_directory,
			                        'flowers/jpg/' + data['image_list'][
				                                            wrong_img_id])
			wrong_img_array = image_processing.load_image_array_flowers(wrong_img_file,
																		img_size)
			wrong_imgs[count, :, :, :] = wrong_img_array

			wrong_classes[count, :] = data['classes'][data['image_list'][
									wrong_img_id]][0:data['n_classes']]

			random_caption = random.randint(0, 4)
			captions[count, :] = \
				data['captions'][data['image_list'][idx]][
								random_caption][0:data['max_caps_len']]

			correct_classes[count, :] = \
				data['classes'][data['image_list'][idx]][
				0:data['n_classes']]
			string_captions = data['str_captions'][data['image_list']
								[idx]][random_caption]

			img_files.append(img_file)
			img_caps.append(string_captions)
			count += 1

		noise_tensor = np.random.uniform(-1, 1, [batch_size, noise_dim])
		return correct_imgs, wrong_imgs, captions, noise_tensor, img_files, \
			   correct_classes, wrong_classes, img_caps, img_ids
	else:
		raise Exception('Failure: No Dataset found')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--z_dim', type=int, default=100,
						help='Dimension used for Noise')

	parser.add_argument('--t_dim', type=int, default=256,
						help='Dimension used for Text feature')

	parser.add_argument('--batch_size', type=int, default=64,
						help='Batch Size')

	parser.add_argument('--image_size', type=int, default=128,
						help='Image Size a, a x a')

	parser.add_argument('--gf_dim', type=int, default=64,
						help='Number of conv2D in the first layer of generator')

	parser.add_argument('--df_dim', type=int, default=64,
						help='Number of conv2D in the first layer of discriminator')

	parser.add_argument('--caption_vector_length', type=int, default=4800,
						help='Length of fixed length caption vectors')

	parser.add_argument('--n_classes', type = int, default = 102,
	                    help = 'Number of classes/class labels of flowers')

	parser.add_argument('--data_dir', type=str, default="Data",
						help='Data Directory')

	parser.add_argument('--learning_rate', type=float, default=0.0002,
						help='Learning Rate')

	parser.add_argument('--beta1', type=float, default=0.5,
						help='Momentum used for Adam Update')

	parser.add_argument('--epochs', type=int, default=500,
						help='Maximum number of epochs')

	parser.add_argument('--save_every', type=int, default=30,
						help='Save Model/Samples every x iterations over '
							 'batches')

	parser.add_argument('--resume_model', type=bool, default=False,
						help='Pre-Trained Model load or not')

	parser.add_argument('--data_set', type=str, default="flowers",
						help='Dat set: MS-COCO, flowers')

	parser.add_argument('--model_name', type=str, default="TAC_GAN",
						help='model_1 or model_2')

	parser.add_argument('--train', type = bool, default = True,
	                    help = 'True while training and otherwise')

	arguments = parser.parse_args()

	model_directory, model_chkpnts_directory, model_samples_directory, model_val_samples_directory,\
							model_summaries_directory = set_directory_paths(arguments)

	datasets_root_directory = join(arguments.data_dir, 'datasets')
	loaded_data = load_trained_data(datasets_root_directory, arguments.data_set,
									arguments.caption_vector_length,
									arguments.n_classes)
	model_choice = {
		'z_dim': arguments.z_dim,
		't_dim': arguments.t_dim,
		'batch_size': arguments.batch_size,
		'image_size': arguments.image_size,
		'gf_dim': arguments.gf_dim,
		'df_dim': arguments.df_dim,
		'caption_vector_length': arguments.caption_vector_length,
		'n_classes': loaded_data['n_classes']
	}

	# Initialize and build the GAN model
	gan_model = model.FlowersGAN(model_choice)
	input_vectors, vars, cal_loss, output_img, checks = gan_model.build()

	disc_optimizer = tf.train.AdamOptimizer(arguments.learning_rate,
									 beta1=arguments.beta1).minimize(cal_loss['disc_loss'],
																	 var_list=vars['d_vars'])
	generator_optimizer = tf.train.AdamOptimizer(arguments.learning_rate,
									 beta1=arguments.beta1).minimize(cal_loss['generator_loss'],
																	 var_list=vars['g_vars'])

	steptensor_global = tf.Variable(1, trainable=False, name='step_global')
	merge_all = tf.summary.merge_all()
	session_tf = tf.InteractiveSession()

	write_summary = tf.summary.FileWriter(model_summaries_directory, session_tf.graph)

	tf.global_variables_initializer().run()
	train_saver = tf.train.Saver(max_to_keep=10000)

	if arguments.resume_model:
		print('Now trying to resume training from previous checkpoint' +
		      str(tf.train.latest_checkpoint(model_chkpnts_directory)))
		if tf.train.latest_checkpoint(model_chkpnts_directory) is not None:
			train_saver.restore(session_tf, tf.train.latest_checkpoint(model_chkpnts_directory))
			print('Model loaded successfully. Training resumed.')
		else:
			print('Failure in loading requested checkpoints. New model in undergoing training.')
	step_global = steptensor_global.eval()
	step_global_assign_op = steptensor_global.assign(step_global)
	for i in range(arguments.epochs):
		batch_count = 0
		while batch_count * arguments.batch_size + arguments.batch_size < \
				loaded_data['data_length']:

			correct_imgs, wrong_imgs, captioned_vectors, noise_tensor, img_files, \
			correct_classes, wrong_classes, img_caps, img_ids = \
							   get_trained_batch(batch_count, arguments.batch_size,
												 arguments.image_size, arguments.z_dim,
	                                              'train', datasets_root_directory,
												 arguments.data_set, loaded_data)

			# DISCR UPDATE
			check_ts = [checks['d_correct_image_loss'], checks['d_wrong_logits_loss'],
			            checks['d_fake_logits_loss'], checks['d_aux_logits_loss'],
			            checks['d_aux_wrong_logits_loss']]

			feed = {
				input_vectors['t_real_image'].name : correct_imgs,
				input_vectors['t_wrong_image'].name : wrong_imgs,
				input_vectors['t_real_caption'].name : captioned_vectors,
				input_vectors['t_z'].name : noise_tensor,
				input_vectors['t_real_classes'].name : correct_classes,
				input_vectors['t_wrong_classes'].name : wrong_classes,
				input_vectors['t_training'].name : arguments.train
			}

			_, disc_loss, gen, d1, d2, d3, d4, d5= session_tf.run([disc_optimizer,
																   cal_loss['disc_loss'], output_img['generator']] + check_ts,
																  feed_dict=feed)

			print("Discriminator total loss: {}\n"
			      "Discriminator cal_loss-1 [Real/Fake cal_loss for real images] : {} \n"
			      "Discriminator cal_loss-2 [Real/Fake cal_loss for wrong images]: {} \n"
			      "Discriminator cal_loss-3 [Real/Fake cal_loss for fake images]: {} \n"
			      "Discriminator cal_loss-4 [Aux Classifier cal_loss for real images]: {} \n"
			      "Discriminator cal_loss-5 [Aux Classifier cal_loss for wrong images]: {}"
			      " ".format(disc_loss, d1, d2, d3, d4, d5))

			# GEN UPDATE
			_, generator_loss, gen = session_tf.run([generator_optimizer, cal_loss['generator_loss'],
													 output_img['generator']], feed_dict=feed)

			# GEN UPDATE TWICE
			_, summary, generator_loss, gen, g1, g2 = session_tf.run([generator_optimizer, merge_all,
																	  cal_loss['generator_loss'], output_img['generator'], checks['g_loss_1'],
																	  checks['g_loss_2']], feed_dict=feed)
			write_summary.add_summary(summary, step_global)
			print("\n\nLOSSES\nDiscriminator Loss: {}\nGenerator Loss: {"
                  "}\nBatch Number: {}\nEpoch: {},\nTotal Batches per "
                  "epoch: {}\n".format( disc_loss, generator_loss, batch_count, i,
                    int(len(loaded_data['image_list']) / arguments.batch_size)))
			print("\nG cal_loss-1 [Real/Fake cal_loss for fake images] : {} \n"
			      "G cal_loss-2 [Aux Classifier cal_loss for fake images]: {} \n"
			      " ".format(g1, g2))
			step_global += 1
			session_tf.run(step_global_assign_op)
			batch_count += 1
			if (batch_count % arguments.save_every) == 0 and batch_count != 0:
				print("Saving Images and Model\n\n")

				save_vis(model_samples_directory, correct_imgs, gen, img_files,
						 img_caps, img_ids)
				saving_path = train_saver.save(session_tf,
                                       join(model_chkpnts_directory,
				                            "latest_model_{}_temp.ckpt".format(
										        arguments.data_set)))

				# Getting a batch for validation
				validate_captions, validate_img_files, validate_img_caps, validate_img_ids = \
                          get_captions_validation(arguments.batch_size, loaded_data,
                                             arguments.data_set, datasets_root_directory)

				shutil.rmtree(model_val_samples_directory)
				os.makedirs(model_val_samples_directory)

				for validate_viz_count in range(0, 4):
					validate_z_noise = np.random.uniform(-1, 1, [arguments.batch_size,
					                                        arguments.z_dim])

					feed_val = {
						input_vectors['t_real_caption'].name : validate_captions,
						input_vectors['t_z'].name : validate_z_noise,
						input_vectors['t_training'].name : True
					}

					val_gen = session_tf.run([output_img['generator']],
											 feed_dict=feed_val)
					save_viz_val(model_val_samples_directory, val_gen,
								 validate_img_files, validate_img_caps,
								 validate_img_ids, arguments.image_size,
								 validate_viz_count)

		# Save the model after every epoch
		if i % 1 == 0:
			epoch_directory = join(model_chkpnts_directory, str(i))
			if not os.path.exists(epoch_directory):
				os.makedirs(epoch_directory)

			saving_path = train_saver.save(session_tf,
			                       join(epoch_directory,
			                            "model_after_{}_epoch_{}.ckpt".
			                                format(arguments.data_set, i)))
			validate_captions, validate_img_files, validate_img_caps, validate_img_ids = \
				  get_captions_validation(arguments.batch_size, loaded_data,
				                     arguments.data_set, datasets_root_directory)

			shutil.rmtree(model_val_samples_directory)
			os.makedirs(model_val_samples_directory)

			for validate_viz_count in range(0, 10):
				validate_z_noise = np.random.uniform(-1, 1, [arguments.batch_size,
				                                        arguments.z_dim])
				feed_val = {
					input_vectors['t_real_caption'].name : validate_captions,
					input_vectors['t_z'].name : validate_z_noise,
					input_vectors['t_training'].name : True
				}
				val_gen = session_tf.run([output_img['generator']], feed_dict=feed_val)
				save_viz_val(model_val_samples_directory, val_gen,
							 validate_img_files, validate_img_caps,
							 validate_img_ids, arguments.image_size,
							 validate_viz_count)


if __name__ == '__main__' :
	main()