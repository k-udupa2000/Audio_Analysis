from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

fs_1, voice_1 = wavfile.read('cdec.wav')  # piano music
fs_2, voice_2 = wavfile.read('Mridanga1.wav') # mridangam

# reducing both signals to same length
voice_1 = voice_1[:, 0]
voice_2 = voice_2[:, 0]
m, = voice_1.shape
voice_2 = voice_2[:m]

figure_1 = plt.figure("Original Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of signal_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, voice_1)
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of signal_2")
plt.plot(np.arange(m)/fs_2, voice_2)
plt.xlabel("Time")
plt.ylabel("Signal")


# mix data
voice = np.c_[voice_1, voice_2]
# artificial mixing for experimentation
A = np.array([[1, 0.5], [0.5, 1]])
X = np.dot(voice, A)


# plotting time domain representation of mixed signal
figure_2 = plt.figure("Mixed Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of mixed signal_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, X[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of mixed signal_2")
plt.plot(np.arange(m)/fs_2, X[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")


# blind source separation using ICA
ica = FastICA()
ica.fit(X)
# get the estimated sources
S_ = ica.transform(X)
# get the estimated mixing matrix
A_ = ica.mixing_
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_) # verifying


# plotting time domain representation of estimated signal
figure_3 = plt.figure("Estimated Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of estimated signal_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, S_[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of estimated signal_2")
plt.plot(np.arange(m)/fs_2, S_[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")

plt.show()


wavfile.write('./separated_1.wav', fs_1, S_[:,0])
wavfile.write('./separated_2.wav', fs_2, S_[:,1])
