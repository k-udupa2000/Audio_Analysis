import scipy.io.wavfile as sw
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift 
from scipy.signal import find_peaks

sample_rate, data = sw.read('output/output1.wav')
_, data2 = sw.read('output/output2.wav')

y = np.abs(fft(data + data2[0: data.shape[0]]))
f = fftfreq(y.shape[0], 1 / sample_rate)
fundamental_freq = np.abs(f[np.argmax(y)])

peaks,_= find_peaks(y,distance=50,height=1000)

print (f[peaks])

plt.plot(f, y)
plt.plot(f[peaks], y[peaks], 'x')
plt.show()
        