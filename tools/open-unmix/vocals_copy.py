import os
import shutil
from tqdm import tqdm
artist_folder = '../../dataset/temp/openunmix_extracted'
os.makedirs('../../dataset/temp/artist20_open_unmix_vocal', exist_ok=True)
artists = [path for path in os.listdir(artist_folder) if os.path.isdir(artist_folder+'/'+path)]

for artist in tqdm(artists):
	artist_path = os.path.join(artist_folder, artist)
	artist_albums = os.listdir(artist_path)
	for album in artist_albums:
		album_path = os.path.join(artist_path, album)
		album_songs = os.listdir(album_path)
		
		for song in album_songs:
			song_path = os.path.join(album_path, song)
			file_path = song_path + "/vocals.wav";
			os.makedirs('../../dataset/temp/artist20_open_unmix_vocal/'+artist+'/'+album, exist_ok=True)
			shutil.move(file_path, '../../dataset/temp/artist20_open_unmix_vocal/'+artist+'/'+album+'/'+song+".wav")
		