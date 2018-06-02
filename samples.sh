#!/bin/bash
list_celebs=`ls voxceleb1/voxceleb1_txt`
for celeb in $list_celebs
do
	echo "Getting samples for: $celeb..."
	celeb_videos=`cat voxceleb1/voxceleb1_txt/$celeb/*.txt | grep '^Youtube' | awk '{print $3;}'`
	for video in $celeb_videos
	do
		if [ -f "voxceleb1/raw_sounds/$video.mp3" ]
		then
		video_samples=`cat "voxceleb1/voxceleb1_txt/$celeb/$video.txt" | grep -v '^[ ,POI,Youtube,Video,Set]' | awk '{print $1;}'`
		for sample in $video_samples
		do
			if [ ! -f "voxceleb1/voxceleb1_txt/$sample.wav" ]
			then
				echo "Sample: $sample..."
				from=`cat "voxceleb1/voxceleb1_txt/$celeb/$video.txt" | grep "^$sample" | awk '{print $2;}'`
				to=`cat "voxceleb1/voxceleb1_txt/$celeb/$video.txt" | grep "^$sample" | awk '{print $3;}'`
				ffmpeg -hide_banner -loglevel panic -i "voxceleb1/raw_sounds/$video.mp3" -ss $from -to $to -c copy "voxceleb1/voxceleb1_txt/$sample.wav"
			fi
		done
		fi
	done
done
