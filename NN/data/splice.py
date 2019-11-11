import os
from os import listdir
from os.path import isfile, join
from getch import getch

gCount = 0        # Global Counter
infolder = "1"    # Folder name to go into
files = [f for f in listdir(infolder) if isfile(join(infolder, f))]

outfolder = infolder + "py"
os.system("mkdir " + outfolder)

from pydub import AudioSegment
from pydub.playback import play

numFiles = len(files)

for p in range(0, numFiles):
  sound = AudioSegment.from_file(infolder + '/' + files[p])
  length = int(len(sound)/1000)
  
  for k in range(1, length):
    chunk = sound[(k-1)*1e3:k*1e3]
    play(chunk)
    print("y/n")
    ans = getch()
    if ans == 'y':
      out_f = open(outfolder + "/%d_crow.mp3" % gCount, 'wb')
      chunk.export(out_f, format='mp3')
    else:
      out_f = open(outfolder + "/%d_NotCrow.mp3" % gCount, 'wb')
      chunk.export(out_f, format='mp3')
    gCount+=1
