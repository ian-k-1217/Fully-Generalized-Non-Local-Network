import src.utility as utility
import src.models as models
from sklearn import manifold
import seaborn as sns
import pandas as pd
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import isfile

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, recall_score, precision_score

def train_model(nb_classes=20,
				slice_length=911,
				artist_folder='../dataset/artist20',
				song_folder='../dataset/melspectrum_artist20_origin',
				melody_folder='../dataset/melody_artist20_origin',
				plots=True,
				train=True,
				load_checkpoint=False,
				save_metrics=True,
				save_valacc_metrics_folder='metrics',
				save_valloss_metrics_folder='metrics',
				save_last_metrics_folder='metrics',
				save_f1_metrics_folder='metrics',
				save_valacc_weights_folder='weights',
				save_valloss_weights_folder='weights',
				save_last_weights_folder='weights',
				save_f1_weights_folder='weights',
				tsne_folder='representation_output',
				batch_size=16,
				nb_epochs=300,
				early_stop=300,
				lr=0.0001,
				album_split=True,
				random_states=42,
				tsne = True):
	"""
	Main function for training the model and testing
	"""

	loss_weights = os.path.join(save_valloss_weights_folder, str(nb_classes) +
						   '_' + str(slice_length) + '_' + str(random_states) + '\\')
	acc_weights = os.path.join(save_valacc_weights_folder, str(nb_classes) +
						   '_' + str(slice_length) + '_' + str(random_states) + '\\')
	last_weights = os.path.join(save_last_weights_folder, str(nb_classes) +
						   '_' + str(slice_length) + '_' + str(random_states) + '\\')
	f1_weights = os.path.join(save_f1_weights_folder, str(nb_classes) +
						   '_' + str(slice_length) + '_' + str(random_states) + '\\')
	tsne_folder = os.path.join(tsne_folder + '\\')
	os.makedirs(save_valloss_weights_folder, exist_ok=True)
	os.makedirs(save_valloss_metrics_folder, exist_ok=True)
	os.makedirs(save_valacc_weights_folder, exist_ok=True)
	os.makedirs(save_valacc_metrics_folder, exist_ok=True)
	os.makedirs(save_last_weights_folder, exist_ok=True)
	os.makedirs(save_last_metrics_folder, exist_ok=True)
	os.makedirs(save_f1_weights_folder, exist_ok=True)
	os.makedirs(save_f1_metrics_folder, exist_ok=True)
	os.makedirs(tsne_folder, exist_ok=True)

	print("Loading dataset...")

	if not album_split:
		# song split
		Y_train, X_train, S_train, M_train, Y_test, X_test, S_test, M_test,\
		Y_val, X_val, S_val, M_val = \
			utility.load_dataset_song_split(song_folder_name=song_folder,
											melody_folder_name=melody_folder,
											artist_folder=artist_folder,
											nb_classes=nb_classes,
											random_state=random_states)
	else:
		Y_train, X_train, S_train, M_train, Y_test, X_test, S_test, M_test,\
		Y_val, X_val, S_val, M_val = \
			utility.load_dataset_album_split(song_folder_name=song_folder,
											 melody_folder_name=melody_folder,
											 artist_folder=artist_folder,
											 nb_classes=nb_classes,
											 random_state=random_states)

	print("Loaded and split dataset. Slicing songs...")

	# Create slices out of the songs
	X_train, Y_train, S_train, M_train = utility.slice_songs(X_train, Y_train, S_train, M_train,
													length=slice_length)
	X_val, Y_val, S_val, M_val = utility.slice_songs(X_val, Y_val, S_val, M_val,
											  length=slice_length)
	X_test, Y_test, S_test, M_test = utility.slice_songs(X_test, Y_test, S_test, M_test,
												 length=slice_length)
	Y_original = Y_test
	print("Training set label counts:", np.unique(Y_train, return_counts=True))
	
	# Encode the target vectors into one-hot encoded vectors
	Y_train, le, enc = utility.encode_labels(Y_train)
	Y_test, le, enc = utility.encode_labels(Y_test, le, enc)
	Y_val, le, enc = utility.encode_labels(Y_val, le, enc)

	# Reshape data as 2d convolutional tensor shape
	X_train = X_train.reshape(X_train.shape + (1,))
	X_val = X_val.reshape(X_val.shape + (1,))
	X_test = X_test.reshape(X_test.shape + (1,))
	
	M_train = M_train.reshape(M_train.shape + (1,))
	M_val = M_val.reshape(M_val.shape + (1,))
	M_test = M_test.reshape(M_test.shape + (1,))

	# build the model
	model = models.model(1, 2, 3, 20, slice_length)
	model.summary()

	# Initialize weights using checkpoint if it exists
	if load_checkpoint:
		print("Looking for previous weights...")
		if isfile(loss_weights):
			print('Checkpoint file detected. Loading weights.')
			#model.load_weights(loss_weights)
			#model = tf.keras.models.load_model(weights)
		else:
			print('No checkpoint file detected.  Starting from scratch.')
	else:
		print('Starting from scratch (no checkpoint)')

	checkpointer_loss = ModelCheckpoint(filepath=loss_weights,
								   verbose=1,
								   save_best_only=True, save_weights_only=True)
	checkpointer_acc = ModelCheckpoint(filepath=acc_weights,
								   monitor='val_accuracy',
								   verbose=1,
								   save_best_only=True, save_weights_only=True)
	checkpointer_f1 = ModelCheckpoint(filepath=f1_weights,
								   monitor='val_f1',
								   mode='max',
								   verbose=1,
								   save_best_only=True, save_weights_only=True)
	earlystopper = EarlyStopping(monitor='val_f1', min_delta=0,
								 patience=early_stop, verbose=0, mode='max')
	
	if song_folder[-1] == 'n':
		type = 'origin'
	elif song_folder[-1] == 'l':
		type = 'vocal'
	print(type)
	split = 'song'
	if album_split:
		split = 'album'
	# Train the model
	if train:
		print("Input Mel-spec Shape", X_train.shape)
		print("Input Melody Shape", M_train.shape)
		history = model.fit(X_train, Y_train, batch_size=batch_size,
							shuffle=True, epochs=nb_epochs,
							verbose=1, validation_data=(X_val, Y_val),
							callbacks=[F1_Metrics(valid_data=(X_val, Y_val)), checkpointer_loss, checkpointer_acc, checkpointer_f1, earlystopper])
		model.save_weights(last_weights)
		if plots:
			utility.plot_history(history)
		#繪出結果
		plot_path = './history/'+str(slice_length)
		os.makedirs(plot_path, exist_ok=True)
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(plot_path + '/' + type + '_' + split + '_' + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states)+"_acc.png")
		plt.close('all')

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(plot_path + '/' + type + '_' + split + '_' + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states)+"_loss.png")
		plt.close('all')
	# Load weights that gave best performance on validation set
	model.load_weights(loss_weights)
	#model = tf.keras.models.load_model(weights)
	filename = os.path.join(save_valacc_metrics_folder, str(nb_classes) + '_'
							+ str(slice_length)
							+ '_' + str(random_states) + '.txt')

	# Score test model
	score = model.evaluate(X_test, Y_test, verbose=0)
	y_score = model.predict(X_test)

	# Calculate confusion matrix
	y_predict = np.argmax(y_score, axis=1)
	y_true = np.argmax(Y_test, axis=1)
	cm = confusion_matrix(y_true, y_predict)

	# Plot the confusion matrix
	class_names = np.arange(nb_classes)
	class_names_original = le.inverse_transform(class_names)
	plt.figure(figsize=(14, 14))
	utility.plot_confusion_matrix(cm, classes=class_names_original,
								  normalize=True,
								  title='Confusion matrix with normalization')
	if save_metrics:
		plt.savefig(filename + '.png', bbox_inches="tight")
	plt.close()
	plt.figure(figsize=(14, 14))

	# Print out metrics
	print('Test score/loss:', score[0])
	print('Test accuracy:', score[1])
	print('\nTest results on each slice:')
	loss_scores = classification_report(y_true, y_predict,
								   target_names=class_names_original)
	loss_scores_dict = classification_report(y_true, y_predict,
										target_names=class_names_original,
										output_dict=True)
	print(loss_scores)

	# Predict artist using pooling methodology
	loss_pooling_scores, loss_pooled_scores_dict = \
		utility.predict_artist(model, X_test, M_test, Y_test, S_test,
							   le, class_names=class_names_original,
							   slices=None, verbose=False)

	# Save metrics
	if save_metrics:
		plt.savefig(filename + '_pooled.png', bbox_inches="tight")
		plt.close()
		with open(filename, 'w') as f:
			f.write("Training mel-spec shape:" + str(X_train.shape))
			f.write("Training melody shape:" + str(M_train.shape))
			f.write('\nnb_classes: ' + str(nb_classes) +
					'\nslice_length: ' + str(slice_length))
			f.write('\nweights: ' + loss_weights)
			f.write('\nlr: ' + str(lr))
			f.write('\nTest score/loss: ' + str(score[0]))
			f.write('\nTest accuracy: ' + str(score[1]))
			f.write('\nTest results on each slice:\n')
			f.write(str(loss_scores))
			f.write('\n\n Scores when pooling song slices:\n')
			f.write(str(loss_pooling_scores))
	
	if tsne:
		print("Modifying model and predicting representation")
		intermed_tensor_func = tf.keras.backend.function([model.layers[0].input],[model.layers[-3].output])
		X_rep = []
		
		# predict representation
		print("Predicting")
		for i in range(len(X_test)):
			X_rep.extend(intermed_tensor_func([np.expand_dims(X_test[i], axis = 0)])[0])
		X_rep = np.array(X_rep)
		# fit tsne
		print("Fitting TSNE {}".format(X_rep.shape))
		tsne_model = manifold.TSNE()
		X_2d = tsne_model.fit_transform(X_rep)

		# save results
		print("Saving results")

		pd.DataFrame({'x0': X_2d[:, 0], 'x1': X_2d[:, 1], 'label': Y_original}).to_csv(tsne_folder + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states) + '_loss_tsne.csv', index=False)

		# save figure
		sns.set_palette("Paired", n_colors=20)
		plt.figure(figsize=(20, 20))
		sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1],
						hue=Y_original, palette=sns.color_palette(n_colors=20))
					
		plt.savefig(tsne_folder + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states) + '_loss_tsne.png', bbox_inches="tight");
	#################
	# Load weights that gave best performance on validation set
	model.load_weights(acc_weights)
	#model = tf.keras.models.load_model(weights)
	filename = os.path.join(save_valacc_metrics_folder, str(nb_classes) + '_'
							+ str(slice_length)
							+ '_' + str(random_states) + '.txt')

	# Score test model
	score = model.evaluate(X_test, Y_test, verbose=0)
	y_score = model.predict(X_test)

	# Calculate confusion matrix
	y_predict = np.argmax(y_score, axis=1)
	y_true = np.argmax(Y_test, axis=1)
	cm = confusion_matrix(y_true, y_predict)

	# Plot the confusion matrix
	class_names = np.arange(nb_classes)
	class_names_original = le.inverse_transform(class_names)
	plt.figure(figsize=(14, 14))
	utility.plot_confusion_matrix(cm, classes=class_names_original,
								  normalize=True,
								  title='Confusion matrix with normalization')
	if save_metrics:
		plt.savefig(filename + '.png', bbox_inches="tight")
	plt.close()
	plt.figure(figsize=(14, 14))

	# Print out metrics
	print('Test score/loss:', score[0])
	print('Test accuracy:', score[1])
	print('\nTest results on each slice:')
	acc_scores = classification_report(y_true, y_predict,
								   target_names=class_names_original)
	acc_scores_dict = classification_report(y_true, y_predict,
										target_names=class_names_original,
										output_dict=True)
	print(acc_scores)

	# Predict artist using pooling methodology
	acc_pooling_scores, acc_pooled_scores_dict = \
		utility.predict_artist(model, X_test, M_test, Y_test, S_test,
							   le, class_names=class_names_original,
							   slices=None, verbose=False)

	# Save metrics
	if save_metrics:
		plt.savefig(filename + '_pooled.png', bbox_inches="tight")
		plt.close()
		with open(filename, 'w') as f:
			f.write("Training mel-spec shape:" + str(X_train.shape))
			f.write("Training melody shape:" + str(M_train.shape))
			f.write('\nnb_classes: ' + str(nb_classes) +
					'\nslice_length: ' + str(slice_length))
			f.write('\nweights: ' + acc_weights)
			f.write('\nlr: ' + str(lr))
			f.write('\nTest score/loss: ' + str(score[0]))
			f.write('\nTest accuracy: ' + str(score[1]))
			f.write('\nTest results on each slice:\n')
			f.write(str(acc_scores))
			f.write('\n\n Scores when pooling song slices:\n')
			f.write(str(acc_pooling_scores))
	
	if tsne:
		print("Modifying model and predicting representation")
		intermed_tensor_func = tf.keras.backend.function([model.layers[0].input],[model.layers[-3].output])
		X_rep = []
		
		# predict representation
		print("Predicting")
		for i in range(len(X_test)):
			X_rep.extend(intermed_tensor_func([np.expand_dims(X_test[i], axis = 0)])[0])
		X_rep = np.array(X_rep)	
		# fit tsne
		print("Fitting TSNE {}".format(X_rep.shape))
		tsne_model = manifold.TSNE()
		X_2d = tsne_model.fit_transform(X_rep)

		# save results
		print("Saving results")

		pd.DataFrame({'x0': X_2d[:, 0], 'x1': X_2d[:, 1], 'label': Y_original}).to_csv(tsne_folder + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states) + '_acc_tsne.csv', index=False)

		# save figure
		sns.set_palette("Paired", n_colors=20)
		plt.figure(figsize=(20, 20))
		sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1],
						hue=Y_original, palette=sns.color_palette(n_colors=20))
					
		plt.savefig(tsne_folder + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states) + '_acc_tsne.png', bbox_inches="tight");
	#################
	
	model.load_weights(last_weights)
	#model = tf.keras.models.load_model(weights)
	filename = os.path.join(save_last_metrics_folder, str(nb_classes) + '_'
							+ str(slice_length)
							+ '_' + str(random_states) + '.txt')

	# Score test model
	score = model.evaluate(X_test, Y_test, verbose=0)
	y_score = model.predict(X_test)

	# Calculate confusion matrix
	y_predict = np.argmax(y_score, axis=1)
	y_true = np.argmax(Y_test, axis=1)
	cm = confusion_matrix(y_true, y_predict)

	# Plot the confusion matrix
	class_names = np.arange(nb_classes)
	class_names_original = le.inverse_transform(class_names)
	plt.figure(figsize=(14, 14))
	utility.plot_confusion_matrix(cm, classes=class_names_original,
								  normalize=True,
								  title='Confusion matrix with normalization')
	if save_metrics:
		plt.savefig(filename + '.png', bbox_inches="tight")
	plt.close()
	plt.figure(figsize=(14, 14))

	# Print out metrics
	print('Test score/loss:', score[0])
	print('Test accuracy:', score[1])
	print('\nTest results on each slice:')
	last_scores = classification_report(y_true, y_predict,
								   target_names=class_names_original)
	last_scores_dict = classification_report(y_true, y_predict,
										target_names=class_names_original,
										output_dict=True)
	print(last_scores)

	# Predict artist using pooling methodology
	last_pooling_scores, last_pooled_scores_dict = \
		utility.predict_artist(model, X_test, M_test, Y_test, S_test,
							   le, class_names=class_names_original,
							   slices=None, verbose=False)

	# Save metrics
	if save_metrics:
		plt.savefig(filename + '_pooled.png', bbox_inches="tight")
		plt.close()
		with open(filename, 'w') as f:
			f.write("Training mel-spec shape:" + str(X_train.shape))
			f.write("Training melody shape:" + str(M_train.shape))
			f.write('\nnb_classes: ' + str(nb_classes) +
					'\nslice_length: ' + str(slice_length))
			f.write('\nweights: ' + last_weights)
			f.write('\nlr: ' + str(lr))
			f.write('\nTest score/loss: ' + str(score[0]))
			f.write('\nTest accuracy: ' + str(score[1]))
			f.write('\nTest results on each slice:\n')
			f.write(str(last_scores))
			f.write('\n\n Scores when pooling song slices:\n')
			f.write(str(last_pooling_scores))

	if tsne:
		print("Modifying model and predicting representation")
		intermed_tensor_func = tf.keras.backend.function([model.layers[0].input],[model.layers[-3].output])
		X_rep = []
		
		# predict representation
		print("Predicting")
		for i in range(len(X_test)):
			X_rep.extend(intermed_tensor_func([np.expand_dims(X_test[i], axis = 0)])[0])
		X_rep = np.array(X_rep)	
		# fit tsne
		print("Fitting TSNE {}".format(X_rep.shape))
		tsne_model = manifold.TSNE()
		X_2d = tsne_model.fit_transform(X_rep)

		# save results
		print("Saving results")

		pd.DataFrame({'x0': X_2d[:, 0], 'x1': X_2d[:, 1], 'label': Y_original}).to_csv(tsne_folder + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states) + '_last_tsne.csv', index=False)

		# save figure
		sns.set_palette("Paired", n_colors=20)
		plt.figure(figsize=(20, 20))
		sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1],
						hue=Y_original, palette=sns.color_palette(n_colors=20))
					
		plt.savefig(tsne_folder + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states) + '_last_tsne.png', bbox_inches="tight");
	#################
	
	model.load_weights(f1_weights)
	#model = tf.keras.models.load_model(weights)
	filename = os.path.join(save_f1_metrics_folder, str(nb_classes) + '_'
							+ str(slice_length)
							+ '_' + str(random_states) + '.txt')

	# Score test model
	score = model.evaluate(X_test, Y_test, verbose=0)
	y_score = model.predict(X_test)

	# Calculate confusion matrix
	y_predict = np.argmax(y_score, axis=1)
	y_true = np.argmax(Y_test, axis=1)
	cm = confusion_matrix(y_true, y_predict)

	# Plot the confusion matrix
	class_names = np.arange(nb_classes)
	class_names_original = le.inverse_transform(class_names)
	plt.figure(figsize=(14, 14))
	utility.plot_confusion_matrix(cm, classes=class_names_original,
								  normalize=True,
								  title='Confusion matrix with normalization')
	if save_metrics:
		plt.savefig(filename + '.png', bbox_inches="tight")
	plt.close()
	plt.figure(figsize=(14, 14))

	# Print out metrics
	print('Test score/loss:', score[0])
	print('Test accuracy:', score[1])
	print('\nTest results on each slice:')
	f1_scores = classification_report(y_true, y_predict,
								   target_names=class_names_original)
	f1_scores_dict = classification_report(y_true, y_predict,
										target_names=class_names_original,
										output_dict=True)
	print(f1_scores)

	# Predict artist using pooling methodology
	f1_pooling_scores, f1_pooled_scores_dict = \
		utility.predict_artist(model, X_test, M_test, Y_test, S_test,
							   le, class_names=class_names_original,
							   slices=None, verbose=False)

	# Save metrics
	if save_metrics:
		plt.savefig(filename + '_pooled.png', bbox_inches="tight")
		plt.close()
		with open(filename, 'w') as f:
			f.write("Training mel-spec shape:" + str(X_train.shape))
			f.write("Training melody shape:" + str(M_train.shape))
			f.write('\nnb_classes: ' + str(nb_classes) +
					'\nslice_length: ' + str(slice_length))
			f.write('\nweights: ' + f1_weights)
			f.write('\nlr: ' + str(lr))
			f.write('\nTest score/loss: ' + str(score[0]))
			f.write('\nTest accuracy: ' + str(score[1]))
			f.write('\nTest results on each slice:\n')
			f.write(str(f1_scores))
			f.write('\n\n Scores when pooling song slices:\n')
			f.write(str(f1_pooling_scores))
			
	if tsne:
		print("Modifying model and predicting representation")
		intermed_tensor_func = tf.keras.backend.function([model.layers[0].input],[model.layers[-3].output])
		X_rep = []
		
		# predict representation
		print("Predicting")
		for i in range(len(X_test)):
			X_rep.extend(intermed_tensor_func([np.expand_dims(X_test[i], axis = 0)])[0])
		X_rep = np.array(X_rep)	
		# fit tsne
		print("Fitting TSNE {}".format(X_rep.shape))
		tsne_model = manifold.TSNE()
		X_2d = tsne_model.fit_transform(X_rep)

		# save results
		print("Saving results")

		pd.DataFrame({'x0': X_2d[:, 0], 'x1': X_2d[:, 1], 'label': Y_original}).to_csv(tsne_folder + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states) + '_f1_tsne.csv', index=False)

		# save figure
		sns.set_palette("Paired", n_colors=20)
		plt.figure(figsize=(20, 20))
		sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1],
						hue=Y_original, palette=sns.color_palette(n_colors=20))
					
		plt.savefig(tsne_folder + str(nb_classes) + '_' + str(slice_length) + '_' + str(random_states) + '_f1_tsne.png', bbox_inches="tight");
	return (loss_scores_dict, loss_pooled_scores_dict, acc_scores_dict, acc_pooled_scores_dict, last_scores_dict, last_pooled_scores_dict, f1_scores_dict, f1_pooled_scores_dict)

class F1_Metrics(tf.keras.callbacks.Callback):
	def __init__(self, valid_data):
		super(F1_Metrics, self).__init__()
		self.validation_data = valid_data
	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
		val_targ = self.validation_data[1]
		if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
			val_targ = np.argmax(val_targ, -1)
		_val_f1 = f1_score(val_targ, val_predict, average='weighted')
		_val_recall = recall_score(val_targ, val_predict, average='weighted')
		_val_precision = precision_score(val_targ, val_predict, average='weighted')
		logs['val_f1'] = _val_f1
		logs['val_recall'] = _val_recall
		logs['val_precision'] = _val_precision
		print(" - val_f1: %f - val_precision: %f - val_recall: %f" % (_val_f1, _val_precision, _val_recall))
		return