#!/bin/bash

Name='Carrion+crow' # Bird Name
Page=1 # Start at the first page
MaxPage=2

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

cd ..
