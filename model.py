
import tensorflow as tf
import tensorflow.contrib.slim as tfslim
from Utils import ops as utilops


class FlowersGAN:

	def __init__(self, command_arguments) :
		self.command_arguments = command_arguments

	def build(self) :

		print('Creating placeholders for the graph')
		image_size_gan = self.command_arguments['image_size']
		correct_image_floats = tf.placeholder('float32', [self.command_arguments['batch_size'], image_size_gan, image_size_gan, 3], name = 'correctImage')
		correct_caption_floats = tf.placeholder('float32', [self.command_arguments['batch_size'], self.command_arguments['caption_vector_length']], name='correctCaptions')
		correct_classes_floats = tf.placeholder('float32', [self.command_arguments['batch_size'], self.command_arguments['n_classes']], name='correctClasses')

		wrong_image_floats = tf.placeholder('float32', [self.command_arguments['batch_size'], image_size_gan, image_size_gan, 3], name = 'wrongImage')
		wrong_classes_floats = tf.placeholder('float32', [self.command_arguments['batch_size'],self.command_arguments['n_classes']], name='wrongClasses')

		noise_floats = tf.placeholder('float32', [self.command_arguments['batch_size'], self.command_arguments['z_dim']], name='inputNoise')
		is_training_bool = tf.placeholder(tf.bool, name='training')

		print('Build Generator')
		gen_fake_image = self.generator(noise_floats, correct_caption_floats, is_training_bool)

		print('Build Discriminator')
		d_correct_image, d_correct_image_logits, d_aux_correct_image, \
		d_aux_correct_image_logits = self.discriminator(correct_image_floats, correct_caption_floats, self.command_arguments['n_classes'], is_training_bool)

		d_wrong_image, d_wrong_image_logits, d_aux_wrong_image, \
		d_aux_wrong_image_logits  = self.discriminator(wrong_image_floats, correct_caption_floats, self.command_arguments['n_classes'], is_training_bool, reuse = True)

		d_fake_images, d_fake_image_logits, d_aux_fake_images, \
		d_aux_fake_image_logits  = self.discriminator(gen_fake_image, correct_caption_floats, self.command_arguments['n_classes'], is_training_bool, reuse = True)

		d_correct_classifications = tf.equal(tf.argmax(d_aux_correct_image, 1), tf.argmax(correct_classes_floats, 1))
		d_correct_image_accuracy = tf.reduce_mean(tf.cast(d_correct_classifications, tf.float32))

		d_wrong_image_classifications = tf.equal(tf.argmax(d_aux_wrong_image, 1),tf.argmax(wrong_classes_floats, 1))
		d_wrong_image_accuracy = tf.reduce_mean(tf.cast(d_wrong_image_classifications,tf.float32))

		d_fake_image_classifications = tf.equal(tf.argmax(d_aux_fake_image_logits, 1), tf.argmax(correct_classes_floats, 1))
		d_fake_image_accuracy = tf.reduce_mean(tf.cast(d_fake_image_classifications, tf.float32))

		tf.get_variable_scope()._reuse = False

		print('Build Loss Function')
		g_fake_image_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_image_logits, labels=tf.ones_like(d_fake_images)))
		g_fake_logits_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_aux_fake_image_logits,labels=correct_classes_floats))

		d_correct_image_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_correct_image_logits, labels=tf.ones_like(d_correct_image)))
		d_aux_logits_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_aux_correct_image_logits,labels=correct_classes_floats))
		d_wrong_logits_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Logits=d_wrong_image_logits, labels=tf.zeros_like(d_wrong_image)))
		d_aux_wrong_logits_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Logits=d_aux_wrong_image_logits,labels=wrong_classes_floats))
		d_fake_logits_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(Logits=d_fake_image_logits,labels=tf.zeros_like(d_fake_images)))

		d_total_loss = d_correct_image_loss + d_aux_logits_loss + d_wrong_logits_loss + d_aux_wrong_logits_loss + d_fake_logits_loss + g_fake_logits_loss
		
		g_total_loss = g_fake_image_loss + g_fake_logits_loss

		t_vars = tf.trainable_variables()
		print('List of all vars')
		for v in t_vars:
			self.add_histogram_summary(v.name, v)

		self.push_scalar_to_tb_summaries(d_total_loss, g_total_loss, d_correct_image_loss, d_wrong_logits_loss, d_fake_logits_loss,
										 d_aux_logits_loss, d_aux_wrong_logits_loss, g_fake_image_loss, g_fake_logits_loss, d_correct_image_accuracy,
										 d_wrong_image_accuracy, d_fake_image_accuracy)

		self.push_to_image_summary('Image from Generator', gen_fake_image, self.command_arguments['batch_size'])

		disc_variables = []
		gen_variables = []

		for disVar in t_vars:
			if 'd_' in disVar.name:
				disc_variables.append(disVar)

		for genVar in t_vars:
			if 'g_' in disVar.name:
				gen_variables.append(genVar)

		input_vectors = {
			'correct_image_floats' : correct_image_floats,
			'wrong_image_floats' : wrong_image_floats,
			'correct_caption_floats' : correct_caption_floats,
			'noise_floats' : noise_floats,
			'correct_classes_floats' : correct_classes_floats,
			'wrong_classes_floats' : wrong_classes_floats,
			'is_training_bool' : is_training_bool,
		}

		variables = {
			'disc_variables' : disc_variables,
			'gen_variables' : gen_variables
		}

		total_gan_loss = {
			'g_total_loss' : g_total_loss,
			'd_total_loss' : d_total_loss
		}

		losses_and_logits = {
			'd_correct_image_loss': d_correct_image_loss,
			'd_wrong_logits_loss': d_wrong_logits_loss,
			'd_fake_logits_loss': d_fake_logits_loss,
			'g_fake_image_loss': g_fake_image_loss,
			'g_fake_logits_loss': g_fake_logits_loss,
			'd_aux_logits_loss': d_aux_logits_loss,
			'd_aux_wrong_logits_loss': d_aux_wrong_logits_loss,
			'd_correct_image_logits': d_correct_image_logits,
			'd_wrong_image_logits': d_wrong_image,
			'd_fake_image_logits': d_fake_image_logits
		}

		return input_vectors, variables, total_gan_loss, gen_fake_image, losses_and_logits


	#Defining generator model
	def generator(self, noise_tensor, embedded_text_tensor, bool_train):

		reduced_embedded_text_tensor = utilops.lrelu(utilops.linear(embedded_text_tensor, self.command_arguments['t_dim'], 'genReducedEmbedding'))
		noise_concat = tf.concat([noise_tensor, reduced_embedded_text_tensor], -1)
		noise_tensor = utilops.linear(noise_concat, self.command_arguments['gf_dim'] * 8 * 8 * 8, 'genNoiseTensor')
		h0 = tf.reshape(noise_tensor, [-1, 8, 8, self.command_arguments['gf_dim'] * 8])
		h0 = tf.nn.relu(tfslim.batch_norm(h0, is_training = bool_train, scope="genFirstLayerInput"))

		h1 = utilops.transpose_conv2d(h0, [self.command_arguments['batch_size'], 16, 16, self.command_arguments['gf_dim'] * 4], name ='genFirstLayerOutput')
		h1 = tf.nn.relu(tfslim.batch_norm(h1, is_training = bool_train, scope="genSecLayerInput"))

		h2 = utilops.transpose_conv2d(h1, [self.command_arguments['batch_size'], 32, 32, self.command_arguments['gf_dim'] * 2], name ='genSecLayerOutput')
		h2 = tf.nn.relu(tfslim.batch_norm(h2, is_training = bool_train, scope="genThirdLayerInput"))
		
		h3 = utilops.transpose_conv2d(h2, [self.command_arguments['batch_size'], 64, 64, self.command_arguments['gf_dim'] * 1], name ='genThirdLayerOutput')
		h3 = tf.nn.relu(tfslim.batch_norm(h3, is_training = bool_train, scope="genFinalLayerInput"))

		h4 = ops.deconv2d(h3, [self.options['batch_size'], 128, 128, 3], name = 'genFinalLayerOutput')

		return (tf.tanh(h3) / 2.0 + 0.5)

	#Defining Dicriminator model
	def discriminator(self, img, embedded_text_tensor, n_classes, bool_train, reuse = False) :
		if reuse :
			tf.get_variable_scope().reuse_variables()

		h0 = utilops.lrelu(utilops.xavier_conv2d(img, self.command_arguments['df_dim'], name ='disConv'))

		h1 = utilops.lrelu(tfslim.batch_norm(utilops.xavier_conv2d(h0, self.command_arguments['df_dim'] * 2, name = 'disFirstConvInput'), reuse=reuse,
										 is_training = bool_train, scope = 'disFirstConvScope'))

		h2 = utilops.lrelu(tfslim.batch_norm(utilops.xavier_conv2d(h1, self.command_arguments['df_dim'] * 4, name = 'disSecConv'), reuse=reuse,
										 is_training = bool_train, scope = 'disSecConvScope'))
		h3 = utilops.lrelu(tfslim.batch_norm(utilops.xavier_conv2d(h2, self.command_arguments['df_dim'] * 8, name = 'disThirdConv'), reuse=reuse,
										 is_training = bool_train, scope = 'disThirdConvScope'))
		h3_shape = h3.get_shape().as_list()
		reduced_embedded_text_tensor = utilops.lrelu(utilops.linear(embedded_text_tensor, self.command_arguments['t_dim'], 'disRedEembedding'))
		reduced_embedded_text_tensor = tf.expand_dims(reduced_embedded_text_tensor, 1)
		reduced_embedded_text_tensor = tf.expand_dims(reduced_embedded_text_tensor, 2)
		tiled_embeddings = tf.tile(reduced_embedded_text_tensor, [1, h3_shape[1], h3_shape[1], 1], name = 'tileEmbeddings')

		h3_concat = tf.concat([h3, tiled_embeddings], 3, name = 'concath3WithTiledEmb')
		h3_new = utilops.lrelu(tfslim.batch_norm(utilops.xavier_conv2d(h3_concat, self.command_arguments['df_dim'] * 8, 1, 1, 1, 1, name = 'disUpdatedThirdConv'), reuse=reuse,
											 is_training = bool_train, scope = 'disFourthConvScope'))

		h3_flat = tf.reshape(h3_new, [self.command_arguments['batch_size'], -1])


		h3 = utilops.linear(h3_flat, 1, 'dish4_lin_rw')
		h3_aux = utilops.linear(h3_flat, n_classes, 'dish4_lin_ac')
		
		return tf.nn.sigmoid(h3), h3, tf.nn.sigmoid(h3_aux), h3_aux


def push_scalar_to_tb_summaries(self, d_total_loss, g_total_loss, d_correct_image_loss, d_aux_logits_loss,
									d_wrong_logits_loss, d_aux_wrong_logits_loss, d_fake_logits_loss, g_fake_image_loss,
									g_fake_logits_loss, d_correctimage_accuracy,
									d_wrong_image_accuracy, d_fake_image_accuracy):

		self.push_scalar_to_summary("D_Loss", d_total_loss)
		self.push_scalar_to_summary("G_Loss", g_total_loss)
		self.push_scalar_to_summary("D loss-1 [Real/Fake loss for real images]", d_correct_image_loss)
		self.push_scalar_to_summary("D loss-2 [Real/Fake loss for wrong images]", d_aux_logits_loss)
		self.push_scalar_to_summary("D loss-3 [Real/Fake loss for fake images]", d_wrong_logits_loss)
		self.push_scalar_to_summary("D loss-4 [Aux Classifier loss for real images]", d_aux_wrong_logits_loss)
		self.push_scalar_to_summary("D loss-5 [Aux Classifier loss for wrong images]", d_fake_logits_loss)
		self.push_scalar_to_summary("G loss-1 [Real/Fake loss for fake images]", g_fake_image_loss)
		self.push_scalar_to_summary("G loss-2 [Aux Classifier loss for fake images]", g_fake_logits_loss)
		self.push_scalar_to_summary("Discriminator Real Image Accuracy", d_correctimage_accuracy)
		self.push_scalar_to_summary("Discriminator Wrong Image Accuracy", d_wrong_image_accuracy)
		self.push_scalar_to_summary("Discriminator Fake Image Accuracy", d_fake_image_accuracy)

	def push_scalar_to_summary(self, sc_name, sc_var):
		with tf.name_scope('summaries'):
			tf.summary.scalar(sc_name, sc_var)

	def add_histogram_summary(self, sc_name, sc_var):
		with tf.name_scope('summaries'):
			tf.summary.histogram(sc_name, sc_var)

	def push_to_image_summary(self, sc_name, sc_var, num_images=1):
		with tf.name_scope('summaries'):
			tf.summary.image(sc_name, sc_var, max_outputs=num_images)