#!/bin/bash

echo "setting up environment..."

DIRECTORY=$(cd `dirname $0` && pwd)

source ~/.profile
workon py3cv4

echo "running recognition script..."
python recognize_faces_video_web.py --encodings family_encodings.pickle --detection-method hog --input "http://192.168.0.33:8080/?action=stream;dummy=param.mjpg"
