import scipy.io.wavfile as sw
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift 
import pandas as pd
from Segmentation import *
import glob
from itertools import groupby 

notes_df = pd.read_csv('notes_freq.csv')
notes = notes_df['Note'].tolist()
freqs = notes_df.Freq.values

audiofiles = glob.glob('audio_notes/*.wav')

for au in audiofiles:
    sample_rate, l = Segment(au, False, False)
    print (au.split('/')[-1]) #[0:4])
    pred = []
    for data in l:  
        
        if data.shape[0] / sample_rate < 0.1:
            continue
        y = fft(data)
        f = fftfreq(y.shape[0], 1 / sample_rate)
        fundamental_freq = np.abs(f[np.argmax(y)])
        index = np.argmin(np.abs(freqs - fundamental_freq))
        pred.append((notes[index], freqs[index]))
     
    pred = [i[0] for i in groupby(pred)] 

    for p in pred:
        print (p[0], p[1])
