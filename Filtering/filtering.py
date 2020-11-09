from numpy import *
from scipy.signal import kaiserord, lfilter, firwin, freqz, iirfilter
from pylab import *
from scipy.io import wavfile
from audio2numpy import open_audio
from matplotlib import pyplot 
from scipy.fftpack import fft,fftfreq

fp = "./Downloads/3.wav"
data, sample_rate = open_audio(fp)
data = data[:,0]

nsamples = len(data)
t = arange(nsamples) / sample_rate

noise2 = np.random.normal(0, .1, data.shape)
noise = 1.5*cos(100*2*pi*t) + 0.5*sin(50*2*pi*t)
noisy_data = data+noise+noise2

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The transition width.
width = 5.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Kaiser window parameters.
N, beta = kaiserord(ripple_db, width)

# The cutoff frequency of the filter.
cutoff_hz = [150,2000.0]

# FIR filter.
taps = firwin(N, divide(cutoff_hz,nyq_rate), window=('kaiser', beta), pass_zero=False)

filtered_x = lfilter(taps, 1.0, noisy_data)

# IIR filter but we are not using it.

# out = signal.iirfilter(17, [150, 2000], ripple_db, btype='band',analog=False, ftype='cheby2', fs, output='out')

# plots.

figure()
plot(taps, 'bo-', linewidth=2)
title('Filter Coefficients (%d taps) from kaiser window' % N)
grid(True)

figure()
w, h = freqz(taps, worN=8000)
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
ylim(-0.05, 1.05)
grid(True)
xlabel('Frequency (Hz)')
ylabel('Gain')
title('Frequency Response')
	
figure()
datafft = fft(data)
fftabs = abs(datafft)
freqs = fftfreq(data.shape[0],1/sample_rate)
plot(freqs,fftabs)
grid(True)
xlabel( 'Frequency (Hz)' )
title('data')

figure()
datafft = fft(noisy_data)
fftabs = abs(datafft)
freqs = fftfreq(data.shape[0],1/sample_rate)
plot(freqs,fftabs)
grid(True)
xlabel( 'Frequency (Hz)' )
title('noisy_data')

figure()
datafft = fft(filtered_x)
fftabs = abs(datafft)
freqs = fftfreq(filtered_x.shape[0],1/sample_rate)
plot(freqs,fftabs)
grid(True)
xlabel( 'Frequency (Hz)' )
title('filtered_data')

show()
wavfile.write("output_1.wav", sample_rate, data)
wavfile.write("output_2.wav", sample_rate, noisy_data)
wavfile.write("output_3.wav", sample_rate, filtered_x)
