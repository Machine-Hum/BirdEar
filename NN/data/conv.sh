#!/bin/bash

for i in *.mp3
do
    sox "$i" "waves/$(basename -s .mp3 "$i").wav"
done
