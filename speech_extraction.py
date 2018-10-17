import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank


frequency_sampling, audio_signal = wavfile.read("hello.wav")
print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
print('Signal duration:', round(audio_signal.shape[0] / 
float(frequency_sampling), 2), 'seconds')
#normalization of audio

audio_signal = audio_signal[:15000]
#print("YG",audio_signal)
features_mfcc = mfcc(audio_signal, frequency_sampling)
print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

filterbank_features = logfbank(audio_signal, frequency_sampling)

var_filter  =  np.square(filterbank_features)
print("filter",filterbank_features)
print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])
filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()