#!/usr/bin/env python
import librosa
import scipy.io.wavfile as wav
from stft_mel import stft_mfcc

""" call stft_mfcc() to execute pre_emphasis(), frame_sig(), fft_sig()
    then finally call mfcc() to get coefficients
"""
#librosa support downsample to 16 Khz
sig, sr = librosa.load('24_crow.wav',mono=True, sr=16000)
#sr,sig = wav.read("24_crow.wav")
print('sam rate', sr)
# only the lower 2-12 of the 40 coefficients are kept.
# frames overlapped 43% consecutive frames
mfcc_feat = stft_mfcc(sig, sr, nfft=512, win_size=0.035, win_stride=0.02, num_ceps=11, nfilt=40)

print('frames shape', mfcc_feat.shape)
print('mfcc', mfcc_feat[1:3,:]) # only show 2 rows in whole arrays(49X11) otherwise too messy
