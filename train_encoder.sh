#!/bin/bash

echo "setting up environment..."

DIRECTORY=$(cd `dirname $0` && pwd)

source ~/.profile
workon py3cv4

echo "running training script..."
python encode_faces.py --dataset family_dataset --encodings family_encodings.pickle --detection-method hog
