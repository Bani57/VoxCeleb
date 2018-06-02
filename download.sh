#!/bin/bash

# PREREQUISITES:
# youtube-dl
# ffmpeg 
#
# USAGE: 
# chmod +x download.sh
# ./download.sh PATH_TO_DATA/list.txt

while IFS='' read -r line || [[ -n "$line" ]]; do
    line=`echo "$line" | sed 's/\r//'`
    if [ ! -f "voxceleb1/raw_sounds/$line.mp3" ]
    then
    	echo "Downloading $line"
    	youtube-dl --extract-audio --audio-format mp3 --audio-quality 1 --output "voxceleb1/raw_sounds/$line.%(ext)s" $line
    	sleep .2
    fi
done < "$1"
