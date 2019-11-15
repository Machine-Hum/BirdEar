# python splice.py <infolder> <outfolder> <birdName> <StartingFileNumber>
import os
import sys
import pdb
from os import listdir
from os.path import isfile, join
from getch import getch

gCount = 0                                                         # Global Counter

infolder = sys.argv[1]                                             # Folder name to go into
outfolder = sys.argv[2]
birdName = sys.argv[3]
startingPoint = int(sys.argv[4])

files = [f for f in listdir(infolder) if isfile(join(infolder, f))]
os.system("mkdir " + outfolder)

from pydub import AudioSegment
from pydub.playback import play

numFiles = len(files)

for p in range(0, numFiles):
  print('Opening file: %s' % files[p])
  sound = AudioSegment.from_file(infolder + '/' + files[p])
  if sound.frame_rate == 44100:  
    length = int(len(sound)/1000)
    
    for k in range(1, length):
      chunk = sound[(k-1)*1e3:k*1e3]
      if gCount > startingPoint:
        play(chunk)
        print(files[p]+':'+ str(gCount) + ", y/n???")
        ans = getch()
        if ans == 'y':
          out_f = open(outfolder + "/%d_%s.mp3" % (gCount, birdName), 'wb')
          chunk.export(out_f, format='mp3')
        else:
          out_f = open(outfolder + "/%d_Not%s.mp3" % (gCount,birdName), 'wb')
          chunk.export(out_f, format='mp3')
      gCount+=1
