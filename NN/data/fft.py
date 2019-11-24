import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq, fftshift
import numpy as np
import json
import pdb
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import ssc
import librosa

folders = ['crow', 'sandpiper']
jsonFname = 'data.json'
bins = 128
root = []

for birds in range(0, len(folders)):
  files = [f for f in listdir(folders[birds]) if isfile(join(folders[birds], f))]
  for l in range(0, len(files)):
    fname = files[l]
    fs, data = wav.read(folders[birds]+'/'+fname)   # Open the file
    if(fs == 44100):
      try:
        data = data[:,0]                                # Remove one of the tracks (convert to mono)
      except:
        continue
      
      mfcc_feat = mfcc(data, fs, nfft=1103).tolist()
      # fbank_feat = logfbank(data, fs)
  
      if "Not" in fname:
        isBird = False
      else:
        isBird = True
  
      data = []
      data = mfcc_feat
  
      root.append({
        'fname'    : fname,
        'fft'      : data,
        'bird'     : folders[birds],
        'isbird'   : isBird,
        'fs'       : fs
      })

with open(jsonFname, 'w') as outfile:
  json.dump(root, outfile, indent=2, sort_keys=True)

