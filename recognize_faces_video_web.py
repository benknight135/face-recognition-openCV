# USAGE
# python recognize_faces_video_web.py --encodings encodings.pickle --input "http://192.168.0.30:8000/video.mjpg"

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import time

import cv2
import numpy as np

import pyttsx3

from imutils.video import FPS
from imutils.video import WebcamVideoStream

# initalise text-to-speech engine
speech_engine = pyttsx3.init()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-s", "--skip_frames" , type=int, default=10,
        help="number of frames to skip")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
stream = WebcamVideoStream(src=args["input"]).start()
fps = FPS().start()
time.sleep(1.0)

if (args["skip_frames"] is not None):
    frames_to_skip = args["skip_frames"]

# loop over frames from the video file stream
while True:
    
    fps.update()

    frame = stream.read()

    if np.shape(frame) != ():
        if frame.shape[0] > 0 and frame.shape[1] > 0:
        
            if args["display"] > 0:
                cv2.imshow("img", frame)
                cv2.waitKey(1)

            #print("resizing for processing...")
            # convert the input frame from BGR to RGB then resize it
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (0,0), fx=0.5, fy=0.5)
            scaling_small = frame_rgb.shape[1] / float(frame_small.shape[1])

            #quick cascade face classifier to find faces in images
            print("finding faces...")
            frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            for (x,y,w,h) in faces:
                #highlight face found
                img = cv2.rectangle(frame_small,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = frame_gray[y:y+h, x:x+w]
                roi_color = frame_small[y:y+h, x:x+w]
                if args["display"] > 0:
                    roi_scaled = cv2.resize(roi_color,(0,0),fx=4,fy=4)
                    cv2.imshow("face", roi_scaled)
                    cv2.waitKey(1)

                #process RIO for better face recognition
                print("finding better faces...")
                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input frame, then compute
                # the facial embeddings for each face
                faces = face_recognition.face_locations(roi_color,
                    model=args["detection_method"])

                print("recognising faces...")
                encodings = face_recognition.face_encodings(roi_color, faces)
                names = []

                print("checking matches...")
                # loop over the facial embeddings
                for encoding in encodings:
                        # attempt to match each face in the input image to our known
                        # encodings
                        matches = face_recognition.compare_faces(data["encodings"],
                                encoding)
                        name = "Unknown"

                        # check to see if we have found a match
                        if True in matches:
                                # find the indexes of all matched faces then initialize a
                                # dictionary to count the total number of times each face
                                # was matched
                                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                                counts = {}

                                # loop over the matched indexes and maintain a count for
                                # each recognized face face
                                for i in matchedIdxs:
                                        name = data["names"][i]
                                        counts[name] = counts.get(name, 0) + 1

                                # determine the recognized face with the largest number
                                # of votes (note: in the event of an unlikely tie Python
                                # will select first entry in the dictionary)
                                name = max(counts, key=counts.get)
                        
                        # update the list of names
                        names.append(name)
                
                # loop over the recognized faces
                for ((top, right, bottom, left), name) in zip(faces, names):

                        # draw the predicted face name on the image
                        cv2.rectangle(roi_color, (left, top), (right, bottom),
                                (0, 255, 0), 1)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(roi_color, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 1)

                        #output voice with name
                        message = "Hello"
                        name_segments = name.split('_')
                        n = 0
                        while n < len(name_segments):
                            message = message + " " + name_segments[n]
                            n += 1

                        print(message)
                        speech_engine.say(message)
                        speech_engine.runAndWait()

                #reset camera stream
                stream = WebcamVideoStream(src=args["input"]).start()

                if args["display"] > 0:
                    roi_scaled = cv2.resize(roi_color,(0,0),fx=4,fy=4)
                    cv2.imshow("face", roi_scaled)
        else:
            print("error: empty image. resetting connection to video feed")
            stream.stop()
            stream = WebcamVideoStream(src=args["input"]).start()
            fps = FPS().start()
            time.sleep(1.0)
    else:
        print("error: empty image resetting connection to video feed")
        stream.stop()
        stream = WebcamVideoStream(src=args["input"]).start()
        fps = FPS().start()
        time.sleep(1.0)
                        
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# close the video file pointers
stream.stop()
fps.stop()
