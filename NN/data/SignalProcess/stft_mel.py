#!/usr/bin/env python

import numpy
from scipy.fftpack import dct

"""Pre-emphasis is a very simple signal processing method
which increases the amplitude of high frequency bands and decrease the amplitudes of lower bands.
"""
def pre_emphasis(signal, preemph=0.97):
    emphasized_signal = numpy.append(signal[0], signal[1:] - preemph * signal[:-1])

    return emphasized_signal

""" split the signal into short-time frames.
    After slicing the signal into frames,
    apply a window function such as the Hamming window to each frame.
    default : win_size=0.025, win_stride=0.01
"""
def frame_sig(emphasized_signal,sr, win_size=0.025, win_stride=0.01):

    frame_length, frame_step = win_size * sr, win_stride * sr  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    frames *= numpy.hamming(frame_length)

    return frames

""" N -point FFT on each frame to calculate the frequency spectrum,
which is also called Short-Time Fourier-Transform (STFT), where N is typically 256 or 512, NFFT = 512
"""
def fft_sig(frames, nfft):
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, nfft))  # Magnitude of the FFT
    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))  # Power Spectrum

    return pow_frames

""" computing filter banks is applying triangular filters, in our case 22 filters,
    on a Mel-scale to the power spectrum to extract frequency bands.
"""
def mfcc(pow_frames,sr,nfft, num_ceps, nfilt):
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sr / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((nfft + 1) * hz_points / sr)

    fbank = numpy.zeros((nfilt, int(numpy.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    #balance the spectrum and improve the Signal-to-Noise (SNR),
    #we can simply subtract the mean of each coefficient from all frames.
    #filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    # Take the Discrete Cosine Transform (DCT) of the given log filterbank energies to give cepstral coefficents.
    ## only the lower 11 of the coefficients are kept.
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep the index from 2-13
    #simplify mfcc
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

    return mfcc

""" call stft_mfcc() to execute pre_emphasis(), frame_sig(), fft_sig()
    then finally call mfcc() to get coefficients
"""
# default : nfft = 256 for downsizing frames
# default : sr=16khz, win_size=0.025, win_stride=0.01,num_ceps =12 and num of filt = 40
# typical frame sizes ranging from 20 ms to 40 ms with 50% (+/-10%)
# overlap between consecutive frames.
def stft_mfcc(sig, sr=16000, nfft=512, win_size=0.025,win_stride=0.01,num_ceps=12, nfilt=40):
    #our wav files are chopped at 1 sec already, which might be too short
    #since the beginning of file may start with silience.
    sig = sig[0:int(1 * sr)]  #Keep the first 1 seconds, wihch affects the frame size

    emphasized_signal = pre_emphasis(sig)

    frames = frame_sig(emphasized_signal,sr, win_size, win_stride)

    pow_frames = fft_sig(frames, nfft)

    mfcc_feat = mfcc(pow_frames, sr,nfft, num_ceps, nfilt)

    return mfcc_feat
