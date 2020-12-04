import numpy as np
import statistics

song_folders = ['melspectrum_artist20_origin', 'melspectrum_artist20_vocal']
split_methods = ['album']
slice_lengths = [94, 157, 313]
summary_files = ['loss_score.csv', 'loss_pooled_score.csv', 'acc_score.csv', 'acc_pooled_score.csv', 'score.csv', 'pooled_score.csv', 'f1_score.csv', 'f1_pooled_score.csv']

for split_method in split_methods:
	for song_folder in song_folders:
		
		album_split = True
		if split_method == 'song':
			album_split = False
		summary_metrics_output_folder = '.\\result\\'+song_folder+'_'+split_method+'_trials\\';
		for slice_len in slice_lengths:
			result_summary_score_avg = [];
			result_summary_score_best = [];
			result_summary_pooled_score_avg = [];
			result_summary_pooled_score_best = [];
			for summary_file in summary_files:
				with open(summary_metrics_output_folder+str(slice_len)+'_'+summary_file,'r') as fp:
					all_lines = fp.readlines();
				data = [];
				for line in all_lines:
					data.append(line.split(','))
			
				print(song_folder, split_method, str(slice_len), summary_file)
				f1_score = [float(data[1][3]), float(data[2][3]), float(data[3][3])]
				print(round(statistics.mean(f1_score), 4), round(max(f1_score), 4))
				
				if summary_file == 'loss_score.csv' or summary_file == 'acc_score.csv' or summary_file == 'score.csv' or summary_file == 'f1_score.csv':
					result_summary_score_avg.append(round(statistics.mean(f1_score), 4))
					result_summary_score_best.append(round(max(f1_score), 4))
				elif summary_file == 'loss_pooled_score.csv' or summary_file == 'acc_pooled_score.csv' or summary_file == 'pooled_score.csv' or summary_file == 'f1_pooled_score.csv':
					result_summary_pooled_score_avg.append(round(statistics.mean(f1_score), 4))
					result_summary_pooled_score_best.append(round(max(f1_score), 4))
					
			print("Frame Level AVG : ", result_summary_score_avg, max(result_summary_score_avg))
			print("Frame Level Best: ", result_summary_score_best, max(result_summary_score_best))
			print("Song Level AVG : ", result_summary_pooled_score_avg, max(result_summary_pooled_score_avg))
			print("Song Level Best: ", result_summary_pooled_score_best, max(result_summary_pooled_score_best))
			
		
