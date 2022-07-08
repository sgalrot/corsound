import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
import torch
import scipy.signal as sps
from torchvision import datasets
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from scipy.io import wavfile
from scipy.signal import butter,filtfilt
from scipy import interpolate, signal
#TODO: change to module import
sys.path.insert(0, './rvad')
from rVAD.rVAD_fast import rVAD_fast
import pathlib
import progressbar

# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, vol. 59, pp. 1-21, 2020. 
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection." 
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.

# Version: 2.0
# 02 Dec 2017, Achintya Kumar Sarkar and Zheng-Hua Tan

# Usage: python rVAD_fast_2.0.py inWaveFile  outputVadLabel

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


if __name__ == "__main__":
    files_list =[]
    source_path=str(sys.argv[1])
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith(".wav"):
                files_list.append(os.path.join(root, file))

  
    progress = progressbar.ProgressBar()
    progress_iter = iter(progress(range(len(files_list))))
    for wav_file_path in files_list:
        progress_iter.next()

        samplerate, signal_wav = wavfile.read(wav_file_path)
        length = signal_wav.shape[0] / samplerate

        if len(signal_wav.shape) > 1:
            signal_wav = signal_wav[:,0]
        new_rate = 16000
        before_vad_filename = "before_vad_decimated.wav"
        number_of_samples = round(len(signal_wav) * float(new_rate) / samplerate)
        signal_wav_decimated = sps.resample(signal_wav, number_of_samples)
        wavfile.write(before_vad_filename, samplerate, signal_wav)


        vad = rVAD_fast(before_vad_filename)    
        vad_length = len(vad)
        time = np.linspace(0., length, signal_wav.shape[0])
        time2 = np.linspace(0., length, vad_length)

        #LPF
        order = 3
        cutoff = 1000
        signal_wav_filtered = butter_lowpass_filter(signal_wav, cutoff, samplerate, order)

        #VAD Cutting
        x = time2
        y = vad
        f = interpolate.interp1d(x, y,kind='nearest')
        xnew = time
        ynew = f(time)   # use interpolation function returned by `nearest`
        signal_wav_after_vad = signal_wav_filtered[np.where(ynew>0)]
        if len(signal_wav_after_vad) == 0:
            continue
         
        #Spectrogram
        f, t, Sxx = signal.spectrogram(signal_wav_after_vad, samplerate, nfft = 16384, nperseg=2048, noverlap=64)
        #TODO: make sure :160 is ~255Hz
        Sxx_to_save = Sxx[:160, :]
        image_resize_shape = [160,160]
        Sxx_to_save = resize(Sxx_to_save, image_resize_shape, anti_aliasing=True)
        Sxx_to_save =  20*np.log(Sxx_to_save)
        Sxx_to_save = img_as_ubyte(Sxx_to_save/np.max(Sxx_to_save))
        
        splitted_path = wav_file_path.split('/')
        write_path = './jpeg/' + '/'.join(splitted_path[-3:])[:-3] + 'jpeg'
        #TODO: change from try except to a function
        try:
            imsave(write_path, Sxx_to_save)
        except:
            path_mkdir = './jpeg/' + '/'.join(splitted_path[-3:-1])
            pathlib.Path(path_mkdir).mkdir(parents=True, exist_ok=True)
            imsave(write_path, Sxx_to_save)

