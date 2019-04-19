#!/bin/bash

echo "setting up environment..."

DIRECTORY=$(cd `dirname $0` && pwd)
echo $DIRECTORY

source ~/.profile
workon py3cv4

echo "running recognition script..."
python $DIRECTORY/recognize_faces_video_web.py --encodings $DIRECTORY/family_encodings.pickle --detection-method hog --input "http://192.168.0.33:8080/?action=stream;dummy=param.mjpg" --rotate 1
