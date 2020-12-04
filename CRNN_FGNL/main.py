# -*- coding: utf-8 -*-

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pandas as pd
import gc

import src.trainer as trainer
import tensorflow as tf
if __name__ == '__main__':
	
	gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
	cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
	print(gpus, cpus)
	gpuid = 0
	tf.config.experimental.set_visible_devices(devices=gpus[gpuid], device_type='GPU')
	tf.config.experimental.set_virtual_device_configuration(
		gpus[gpuid],
		[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)]
	)
	
	'''
	1s 32 frames
	3s 94 frames
	5s 157 frames
	6s 188 frames
	10s 313 frames
	20s 628 frames
	29.12s 911 frames
	'''
	song_folders = ['melspectrum_artist20_origin', 'melspectrum_artist20_vocal']
	split_methods = ['album']
	slice_lengths = [94, 157, 313]
	random_state_list = [0, 21, 42]
	train_bool = True;
	for split_method in split_methods:
		for song_folder in song_folders:
			album_split = True
			if split_method == 'song':
				album_split = False
			##=========================================================
			summary_metrics_output_folder = '.\\result\\'+song_folder+'_'+split_method+'_trials'
			for slice_len in slice_lengths:

				loss_scores = []
				acc_scores = []
				last_scores = []
				f1_scores = []
				loss_pooling_scores = []
				acc_pooling_scores = []
				last_pooling_scores = []
				f1_pooling_scores = []
				for i in range(len(random_state_list)):
					loss_score, loss_pooling_score, acc_score, acc_pooling_score, last_score, last_pooling_score, f1_score, f1_pooling_score = trainer.train_model(
						nb_classes=20,
						slice_length=slice_len,
						song_folder='../dataset/'+song_folder,
						lr=0.001,
						train=train_bool,
						load_checkpoint=True,
						plots=False,
						album_split=album_split,
						random_states=random_state_list[i],
						save_metrics=True,
						save_valacc_metrics_folder='.\\result\\'+song_folder+'_'+split_method+'_acc_metrics',
						save_valloss_metrics_folder='.\\result\\'+song_folder+'_'+split_method+'_loss_metrics',
						save_last_metrics_folder='.\\result\\'+song_folder+'_'+split_method+'_last_metrics',
						save_f1_metrics_folder='.\\result\\'+song_folder+'_'+split_method+'_f1_metrics',
						save_valacc_weights_folder='.\\result\\'+song_folder+'_'+split_method+'_acc_weights',
						save_valloss_weights_folder='.\\result\\'+song_folder+'_'+split_method+'_loss_weights',
						save_last_weights_folder='.\\result\\'+song_folder+'_'+split_method+'_last_weights',
						save_f1_weights_folder='.\\result\\'+song_folder+'_'+split_method+'_f1_weights',
						tsne = False)
					

					loss_scores.append(loss_score['weighted avg'])
					acc_scores.append(acc_score['weighted avg'])
					last_scores.append(last_score['weighted avg'])
					f1_scores.append(f1_score['weighted avg'])
					loss_pooling_scores.append(loss_pooling_score['weighted avg'])
					acc_pooling_scores.append(acc_pooling_score['weighted avg'])
					last_pooling_scores.append(last_pooling_score['weighted avg'])
					f1_pooling_scores.append(f1_pooling_score['weighted avg'])
					gc.collect()

				os.makedirs(summary_metrics_output_folder, exist_ok=True)

				pd.DataFrame(loss_scores).to_csv(
					'{}/{}_loss_score.csv'.format(summary_metrics_output_folder, slice_len))
					
				pd.DataFrame(loss_pooling_scores).to_csv(
					'{}/{}_loss_pooled_score.csv'.format(
						summary_metrics_output_folder, slice_len))
				
				pd.DataFrame(acc_scores).to_csv(
					'{}/{}_acc_score.csv'.format(summary_metrics_output_folder, slice_len))

				pd.DataFrame(acc_pooling_scores).to_csv(
					'{}/{}_acc_pooled_score.csv'.format(
						summary_metrics_output_folder, slice_len))
						
				pd.DataFrame(last_scores).to_csv(
					'{}/{}_score.csv'.format(summary_metrics_output_folder, slice_len))

				pd.DataFrame(last_pooling_scores).to_csv(
					'{}/{}_pooled_score.csv'.format(
						summary_metrics_output_folder, slice_len))
						
				pd.DataFrame(f1_scores).to_csv(
					'{}/{}_f1_score.csv'.format(summary_metrics_output_folder, slice_len))

				pd.DataFrame(f1_pooling_scores).to_csv(
					'{}/{}_f1_pooled_score.csv'.format(
						summary_metrics_output_folder, slice_len))
