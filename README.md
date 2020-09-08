# Positions, Channels, and Layers: Fully Generalized Non-Local Network for Singer Identification
This code is an implementation for the paper 
"[Positions, Channels, and Layers: Fully Generalized Non-Local Network for Singer Identification](https://)".
The code is developed based on the [TensorFlow 2.1](https://www.tensorflow.org/) framework.
![image](https://github.com/ian-k-1217/Fully-Generalized-Non-Local-Network/blob/master/images/FGNL_Fig1.png)

## License
The code and the models in this repo are released under the [CC-BY-NC 4.0 LICENSE](https://github.com/i-yuan-kuo/Fully-Generalized-Non-Local-Network/blob/master/LICENSE).

## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@article{Fully Generalized Non Local 2021,
  author =   {},
  title =    {Positions, Channels, and Layers: Fully Generalized Non-Local Network for Singer Identification},
  conference =  {Submitted to AAAI},
  year =     {2021}
}
```
## Dependency
The code is written in Python 3.7 and is built on the Python packages, including (but not limited to):
- dill==0.3.1.1
- h5py==2.10.0
- librosa==0.7.0
- llvmlite==0.29.0
- matplotlib==3.1.1
- numpy==1.17.2
- numba==0.45.1
- opencv-contrib-python==3.4.2.17
- opencv-python==4.1.0.25
- pandas==0.25.1
- scikit-learn==0.21.3
- scipy==1.4.1
- seaborn==0.10.1
- tensorflow-gpu==2.1.0


Batch installation is possible using the supplied "requirements.txt" with pip or conda.

````cmd
pip install -r requirements.txt
````

Additional installation details (recommended for replication and strong performance):
- OS: Windows 10 Pro
- Python: 3.7.7
- GPU: Nvidia TitanRTX
- CUDA: 10.1
- CUDNN: 10.1
- [ffmpeg](http://ffmpeg.org/download.html) is required by Librosa to convert audio files into spectrograms. 

## Datasets
- **Preprocess the dataset.**
1. Download the dataset [artist20](https://labrosa.ee.columbia.edu/projects/artistid/) and extract it into "*dataset/artist20*". (Total 1413 mp3 music files)
2. Convert those .mp3 file format to .wav into "*dataset/artist20_wav*". (Folder structure should follow artist20's)

- **Prepare the Vocal-only dataset with [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch).**
1. Download the tool "Open-Unmix" and extract it into "*tools/open-unmix*". (You can see the files *output.py* and *vocals_copy.py* is placed with the files from Open-Unmix in this folder)
2. Run "*output.py*" in "*tools/open-unmix*". (It will seperate the vocal from the .wav which is generated at the step 1.)
3. Run "*vocals_copy.py*" in "*tools/open-unmix*". (It will place all the seperated files into "*dataset/temp/artist20_open_unmix_vocal*")

- **Extract the Mel-Spectrum and Melody.**
1. Install melody generating tool [CREPE](https://github.com/marl/crepe).
````cmd
pip install crepe
````
2. Run "*tools/extract_melspectrum.py*" to generate the Mel-Spectrum. (Data will be generated in "*dataset/melspectrum_artist20_origin*" and "*dataset/melspectrum_artist20_vocal*")
3. Run "*tools/extract_melody.py*" to generate the Melody. (Data will be generated in "*dataset/melody_artist20_origin*" and "*dataset/melody_artist20_vocal*")

- **Remove unused files** - Remove the folder "*dataset/temp*", "*dataset/artist20_wav*".

## Usage
- **Prepare Datasets** - Follow the [Datasets](#Datasets) section to download artist20 and generate the datasets of Mel-Spectrum and Melody.
- **Train and Evaluate *(Comming Soon)*** - Run *main.py*, it will begin a training loop which runs three independent trials for each audio length in {3s, 5s, 10s} on the datasets "Origin" and "Vocal-only".
- **Representation Visualization *(Comming Soon)*** - You can modify the parameter *tsne* to *True* in *main.py*, it will generate the t-sne image during the training and evaluating period.

## Results and Comparisons

*Evaluation results on the benchmark [artist20](https://labrosa.ee.columbia.edu/projects/artistid/) dataset.*

- **Table 1:** *The average and best F1 scores of frame and song levels in various length settings (3 runs). Bold is the comparison winner of the same series (CRNN or CRNNM) model.*

![image](https://github.com/ian-k-1217/Fully-Generalized-Non-Local-Network/blob/master/images/Result_Table1.png)

- **Table 2:** *Ablation experiments of CRNN with three attention modules, including NL (Wang et al. 2018), FGNL_LIGHT, and FGNL. Bold indicates the comparison winner of the model.*

![image](https://github.com/ian-k-1217/Fully-Generalized-Non-Local-Network/blob/master/images/Result_Table2.png)

- **Table 3:** *Ablation experiments of CRNN_FGNL with and without Gaussian smoothing, MoSE, and SE (Hu, Shen, and Sun 2018) mechanisms. Bold indicates the comparison winner of the model.*

![image](https://github.com/ian-k-1217/Fully-Generalized-Non-Local-Network/blob/master/images/Result_Table3.png)

- **Representation Visualization:** *Visualization of the embeddings (projected into 2-D space by t-SNE) under the original audio file setting of the 5-sec frame level test samples. From left to right are CRNN, CRNN_NL, CRNN_FGNL_LIGHT, and CRNN_FGNL.*

![image](https://github.com/ian-k-1217/Fully-Generalized-Non-Local-Network/blob/master/images/Result_RepresentationVisualization.png)

## Demo

- **The best viewing experience is to set the player (the option in the lower right corner) to 1080p.**

[![Watch the video](https://github.com/ian-k-1217/Fully-Generalized-Non-Local-Network/blob/master/images/FGNL_Demo.png)](https://drive.google.com/file/d/1N_-N78TeibKLTB7Kj-wTw7YjhMlTuQrz/preview)
