import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from audio2numpy import open_audio
from scipy.io.wavfile import write
from scipy import signal
from scipy.signal import hilbert
import math

def low_pass_filter(sig, Cutoff):
    N  = 3    # Filter order
    Wn = Cutoff # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B,A, sig)
    return smooth_data



def get_tone_arrays(break_points, time, samplerate, sig):
    tones = []
    for i in range(len(break_points) - 1):
        tone = sig[break_points[i]:break_points[i + 1]]
        tones.append(tone)
    # Appending last tone.
    tones.append(sig[break_points[-1]:time[-1]*samplerate])
    return tones
    
def writeToAudioFile(tones, samplerate):
    for i in range(len(tones)):
        filename = './output/' 'output' + str(i + 1) + '.wav'
        write(filename, samplerate, tones[i])


def getNearPoints(arr, ind, n, samplerate):
    if ind >= n:
        return []
    l = [arr[ind]]
    # print(ind)
    count = 0
    for i in range(ind + 1, n):
        if count < 1:
            if (arr[i] - l[-1])/samplerate > 0.2:
                return l
            l.append(arr[i])
            count += 1
        else:
            break
    return l