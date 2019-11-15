import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq, fftshift
import numpy as np
import json

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
      X = fft(data)                                   # Take FFT
      freqs = fftfreq(len(data)) * fs
      freqs = freqs[0:int(len(freqs)/2)]
      X = X[0:int(len(X)/2)]                          # Remove the negative bins
      X = abs(X)                                      # Remove the j shit
      spacing = int(len(X) / bins)                    # Requires spacing
  
      k = list()
      for i in range(0,bins):
        k.append(sum(X[i*spacing:(i+1)*spacing]))
  
      k = k/max(k)
      k = k*256;
      k = [int(i) for i in k]
  
      if "Not" in fname:
        isBird = False
      else:
        isBird = True
  
      data = []
      data = k
  
      root.append({
        'fname'    : fname,
        'fft'      : data,
        'bird'     : folders[birds],
        'isbird'   : isBird,
        'fs'       : fs
      })

with open(jsonFname, 'w') as outfile:
  json.dump(root, outfile, indent=2, sort_keys=True)

