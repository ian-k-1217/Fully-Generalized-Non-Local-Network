import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

def model(frequency_axis = 1, time_axis = 2, channel_axis = 3, class_num = 20, slice_length = 157):
	## mel-spectrogram cnn branch
	# shape of input data (frequency, time, channels)
	mel_input = tf.keras.layers.Input(shape=(128, slice_length, 1));
	x = tf.keras.layers.BatchNormalization(axis = frequency_axis)(mel_input);
	cb1 = conv_block(x, 64, kernel_size = (3, 3), activation = 'elu', pool_size = (2, 2), channel_axis = 3, block_name = 'MelBranch_Conv_1', pooling = False);
	cb2 = conv_block(cb1, 128, kernel_size = (3, 3), activation = 'elu', pool_size = (2, 2), channel_axis = 3, block_name = 'MelBranch_Conv_2', pooling = True);
	cb3 = conv_block(cb2, 128, kernel_size = (3, 3), activation = 'elu', pool_size = (2, 2), channel_axis = 3, block_name = 'MelBranch_Conv_3', pooling = True);
	cb4 = conv_block(cb3, 128, kernel_size = (3, 3), activation = 'elu', pool_size = (2, 2), channel_axis = 3, block_name = 'MelBranch_Conv_4', pooling = True);
	
	x = FullyGeneralizedNonLocalLayer(cb4, [cb4, cb3]);
	
	####========================================================================================
	## reshape
	# (frequency, time, channels) => (time, frequency, channel)
	x = tf.keras.layers.Permute((time_axis, frequency_axis, channel_axis))(x);
	x = tf.keras.layers.Reshape((-1, x.shape[2] * x.shape[3]))(x);
	
	## recurrent layer
	x = tf.keras.layers.GRU(units = 32, return_sequences = True)(x);
	x = tf.keras.layers.GRU(units = 32, return_sequences = False)(x);
	x = tf.keras.layers.Dropout(rate = 0.3)(x);

	## output layer
	x = tf.keras.layers.Dense(class_num, kernel_regularizer = tf.keras.regularizers.l2(0.01), activity_regularizer = tf.keras.regularizers.l1(0.01))(x);
	y = tf.keras.layers.Activation(activation = 'softmax')(x);	
	
	####========================================================================================
	model = tf.keras.Model(inputs = (mel_input) , outputs = y);
	
	adam = tf.keras.optimizers.Adam(lr = 0.0001);
	model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy']);
	
	return model;


## convolution block module
def conv_block(tensor_input, filters = 64, kernel_size = (3, 3), activation = 'elu', pool_size = (2, 2), channel_axis = 3, block_name = 'Conv', pooling = True):
	x = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', name=block_name+'_Conv')(tensor_input);
	x = tf.keras.layers.Activation(activation, name=block_name+'_Activation')(x);
	x = tf.keras.layers.BatchNormalization(axis=channel_axis, name=block_name+'_BN')(x);
	if pooling:
		x = tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = pool_size, name=block_name+'_Pooling')(x);
	x = tf.keras.layers.Dropout(rate = 0.1, name=block_name+'_Dropout')(x);
	
	return x;

## Fully Generalized Non-Local Layer
def FullyGeneralizedNonLocalLayer(theta_input, phi_g_inputs, channel_denominator = 32):
	## theta branch [Eq. 2]
	theta = tf.keras.layers.Conv2D(filters = int(theta_input.shape[3] / channel_denominator), kernel_size = (1, 1))(theta_input);
	theta = Gaussian_Filter_Block(theta);
	theta = tf.keras.layers.Reshape((theta.shape[1] * theta.shape[2], theta.shape[3]))(theta);
	
	## phi branch [Eq. 3]
	phi_branches = [];
	for phi_input in phi_g_inputs:
		phi_branches.append(FGNL_PhiGBranch(phi_input, channel_denominator));
	
	if len(phi_branches) == 1:
		phi = phi_branches[0];
	else:
		phi = tf.keras.layers.concatenate(phi_branches, axis=1);
	
	## g branch [Eq. 4]
	g_branches = [];
	for g_input in phi_g_inputs:
		g_branches.append(FGNL_PhiGBranch(g_input, channel_denominator));
	
	if len(g_branches) == 1:
		g = g_branches[0];
	else:
		g = tf.keras.layers.concatenate(g_branches, axis=1);
	
	## Roll, MoSE, residual connection
	R = FGNL_Roll(theta, phi, g);		# [Eq. 7: r([G(phi_i(Xi)), G(phi_j(Xj))])]
	reweighted_R = MoSE_Block(R);					# [Eq. 8]
	Y = tf.keras.layers.Reshape((theta_input.shape[1], theta_input.shape[2], int(theta_input.shape[3] / channel_denominator) * reweighted_R.shape[3]))(reweighted_R);
	Y = tf.keras.layers.Conv2D(filters = theta_input.shape[3], kernel_size = (1, 1))(Y);
	Z = tf.keras.layers.add([theta_input, Y]);		# [Eq. 9]
	
	return Z;
	
## Rolling of FGNL:  r([G(phi_i(Xi)), G(phi_j(Xj))])
def FGNL_Roll(theta, phi, g):
	R = [];
	for phi_channel_index in range(phi.shape[2]):
		fi = tf.keras.layers.dot([theta, tf.roll(phi, phi_channel_index, axis=2)], axes = 2);
		fi = tf.keras.layers.Softmax(axis=1)(fi);
		Yk = tf.keras.layers.dot([fi, g], axes = [2, 1]);
		R.append(tf.keras.layers.Reshape((Yk.shape[1], Yk.shape[2], 1))(Yk));
	
	R = tf.keras.layers.concatenate(R, axis = -1);
	return R;
	
## MoSE Block
def MoSE_Block(R_input):
	w = tf.keras.layers.GlobalAveragePooling2D()(R_input);
	w = tf.keras.layers.Reshape((1, w.shape[1], 1))(w);
	w = tf.keras.layers.Conv2D(filters = 16, kernel_size = (1, 1))(w);
	w = tf.keras.layers.Activation(activation = 'relu')(w);
	w = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1, 1))(w);
	w = tf.keras.layers.Flatten()(w);
	w = tf.keras.layers.Activation(activation = 'softmax')(w);
	w = tf.keras.layers.Reshape((1, 1, w.shape[1]))(w);
	reweighted_R = tf.keras.layers.Multiply()([w, R_input]);
	
	return reweighted_R;

## Input of FGNL's Phi G Branch	
def FGNL_PhiGBranch(tensor_input, channel_denominator = 32):
	x = tf.keras.layers.Conv2D(filters = int(tensor_input.shape[3] / channel_denominator), kernel_size = (1, 1))(tensor_input);
	x = Gaussian_Filter_Block(x);
	x = tf.keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x);
	return x;


## Gaussian Smoothing Filter:  G()
def Gaussian_Filter(shape, dtype=None, sigma = 0.707):
	kernel_temp = cv2.getGaussianKernel(shape[0], sigma);
	kernel_temp = np.expand_dims(np.expand_dims(np.outer(kernel_temp, kernel_temp.transpose()), axis=-1), axis=-1);
	kernel = kernel_temp;
	for i in range(shape[2] - 1):
		kernel = np.concatenate((kernel, kernel_temp), axis = 2);
		
	return tf.keras.backend.variable(kernel, dtype='float32');
		
def Gaussian_Filter_Block(tensor_input):	
	return tf.keras.layers.DepthwiseConv2D(kernel_size = (5, 5), strides = 1, padding="same", use_bias=False, depthwise_initializer = Gaussian_Filter, trainable = False)(tensor_input);
