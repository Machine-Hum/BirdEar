#!/bin/bash

Name='sandpiper' # Bird Name
Page=2 # Start at the first page

wget "https://www.xeno-canto.org/explore?query=${Name}&pg=${Page}"

fname="explore?query=${Name}&pg=${Page}"
cat ${fname} | grep download | grep -o -P "href.*download'" | tr -d "href='" > out
rm ${fname}

mkdir $Page
mv out $Page
cd $Page

for i in {1..30}
do
  dl="$(sed -n "${i}p" out)"
  echo $dl
  wget "https://www.xeno-canto.org${dl}"
done

rm out

cd ..
