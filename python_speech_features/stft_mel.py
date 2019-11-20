#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import ssc
import scipy.io.wavfile as wav
import numpy
import librosa

#librosa support downsample
sig, rate = librosa.load('24_crow.wav',mono=True, sr=8000) # Downsample 22050 Hz to 8kHz for STFT and Mel
#(rate,sig) = wav.read("english.wav")

#our wav files are chopped at 1 sec already, which might be too short
#since the beginning of file may start with silience.
sig = sig[0:int(1 * rate)]  #Keep the first 1 seconds, wihch affects the frame size

#call rfft internally
mfcc_feat = mfcc(sig, rate, winfunc=numpy.hamming)
print('mfcc', mfcc_feat[1:3,:]) # only show 2 rows in whole arrays(99X13) otherwise too messy

d_mfcc_feat = delta(mfcc_feat, 2)
print('delta_mfcc', d_mfcc_feat[1:3,:])

#call rfft internally
fbank_feat = logfbank(sig, rate, winfunc=numpy.hamming)
print('fbank ', fbank_feat[1:3,:])#arrays(99X26)

ssc_feat = ssc(sig, rate, winfunc=numpy.hamming)
