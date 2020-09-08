import os
from tqdm import tqdm
import multiprocessing
def music_unmix(music, output_path):
	os.system('python test.py "'+music+'" --outdir '+output_path)
if __name__ == '__main__':
	pool = multiprocessing.Pool(4)
	
	artist_folder = '../../dataset/artist20_wav'
	os.makedirs('../../dataset/temp/openunmix_extracted', exist_ok=True)
	artists = [path for path in os.listdir(artist_folder) if os.path.isdir(artist_folder+'/'+path)]

	for artist in tqdm(artists):
		artist_path = os.path.join(artist_folder, artist)
		artist_albums = os.listdir(artist_path)
		for album in artist_albums:
			album_path = os.path.join(artist_path, album)
			album_songs = os.listdir(album_path)
			
			for song in album_songs:
				song_path = os.path.join(album_path, song)
				if not os.path.isdir(os.path.join(os.path.join(os.path.join('../../dataset/temp/openunmix_extracted', artist), album), os.path.splitext(song)[0])):
					print(os.path.join(os.path.join(os.path.join('../../dataset/temp/openunmix_extracted', artist), album), os.path.splitext(song)[0]))
					pool.apply_async(music_unmix, (song_path, os.path.join(os.path.join(os.path.join('../../dataset/temp/openunmix_extracted', artist), album), os.path.splitext(song)[0])))
					

	'''

	'''
	print("--" * 10);
	pool.close();
	pool.join();
	print("All process done.");
	#os.system('python vocals_copy.py')