# import the necessary packages
from pyimagesearch.tempimage import TempImage
from dropbox.client import DropboxOAuth2FlowNoRedirect
from dropbox.client import DropboxClient
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import numpy as np
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
args = vars(ap.parse_args())
 
# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

if conf["use_dropbox"]:
	# connect to dropbox and start the session authorization process
	flow = DropboxOAuth2FlowNoRedirect(conf["dropbox_key"], conf["dropbox_secret"])
	print "[INFO] Authorize this application: {}".format(flow.start())
	authCode = raw_input("Enter auth code here: ").strip()
 
	# finish the authorization and grab the Dropbox client
	(accessToken, userID) = flow.finish(authCode)
	client = DropboxClient(accessToken)
	print "[SUCCESS] dropbox account linked"

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))
 
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print "[INFO] warming up..."
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# configuration for split frame
frame_split = conf["frame_split"]

(width, _) = tuple(conf["resolution"])
if width%frame_split != 0:
        frame_split = frame_split + width%frame_split
        
index = (frame_split / 2) - frame_split
list_frame_split = {index - 1: -1, index: 0, index + frame_split: width, index + frame_split + 1: width + 1}
index = index + 1

i = width / frame_split
while i < width:
        list_frame_split[index] = i
        i = i + (width / frame_split)
        index = index + 1

print(list_frame_split)

# configuration for faces detecting
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# configuration for object tracking
track = False

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image and initialize
	# the timestamp and occupied/unoccupied text
	frame = f.array
	timestamp = datetime.datetime.now()
	text = "Unoccupied"

	if not track:

                # draw a line for vertically split frame
        ##	for key_line in list_frame_split:
        ##                if list_frame_split[key_line] == (frame.shape[1] / 2):
        ##                        cv2.line(frame, (list_frame_split[key_line], 0), (list_frame_split[key_line], frame.shape[0]), (0, 255, 0), 2)
        ##                else:
        ##                        cv2.line(frame, (list_frame_split[key_line], 0), (list_frame_split[key_line], frame.shape[0]), (224, 224, 224), 1)
         
                # resize the frame, convert it to grayscale, and blur it
                #frame = imutils.resize(frame, width=500)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
         
                # if the average frame is None, initialize it
                if avg is None:
                        print "[INFO] starting background model..."
                        avg = gray.copy().astype("float")
                        rawCapture.truncate(0)
                        continue
         
                # accumulate the weighted average between the current frame and
                # previous frames, then compute the difference between the current
                # frame and running average
                cv2.accumulateWeighted(gray, avg, 0.5)
                frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

                # threshold the delta image, dilate the thresholded image to fill
                # in holes, then find contours on thresholded image
                thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
                        cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         
                # loop over the contours
                biggest = None
                for c in cnts:
                        # if the contour is too small, ignore it
                        if cv2.contourArea(c) < conf["min_area"] or cv2.contourArea(c) > conf["max_area"]:
                                continue

                        if biggest == None or cv2.contourArea(biggest) < cv2.contourArea(c):
                                biggest = c
                                
                        # compute the bounding box for the contour, draw it on the frame,
                        # and update the text
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        text = "Occupied"

                        # camshift (object tracking)
                        # setup initial location of window
                        track_window = (x, y, w, h)
                        # set up the ROI for tracking
                        roi = frame[y:y+h, x:x+w]
                        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

                        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
                        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                        track = True
                        
                #compute position of biggest contour
                if biggest != None :
                        (x, _, w, _) = cv2.boundingRect(biggest)
                        position = x + (w / 2)
        ##                print("Position: " + str(position))
                        for key_line in list_frame_split:
                                if position >= list_frame_split[key_line] and position < list_frame_split[(key_line + 1)]:
        ##                                print("---------")
        ##                                print("Between " + str(list_frame_split[(key_line)]) + " and " + str(list_frame_split[(key_line + 1)]))
        ##                                print("Between " + str(key_line) + " and " + str(key_line + 1))
        ##                                print("---------")
                                        cv2.putText(frame, "POSITION:" + str(key_line), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                        break

        else :
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
                # apply meanshift to get the new location
                track, track_window = cv2.CamShift(dst, track_window, term_crit)
                   
                # Draw it on image
                pts = cv2.boxPoints(track)
                pts = np.int0(pts)
                frame = cv2.polylines(frame,[pts],True, 255,2)
                                
        # draw the text and timestamp on the frame
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

##	# check to see if the room is occupied
##	if text == "Occupied":
##		# check to see if enough time has passed between uploads
##		if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
##			# increment the motion counter
##			motionCounter += 1
## 
##			# check to see if the number of frames with consistent motion is
##			# high enough
##			if motionCounter >= conf["min_motion_frames"]:
##				# check to see if dropbox sohuld be used
##				if conf["use_dropbox"]:
##					# write the image to temporary file
##					t = TempImage()
##					cv2.imwrite(t.path, frame)
## 
##					# upload the image to Dropbox and cleanup the tempory image
##					print "[UPLOAD] {}".format(ts)
##					path = "{base_path}/{timestamp}.jpg".format(
##						base_path=conf["dropbox_base_path"], timestamp=ts)
##					client.put_file(path, open(t.path, "rb"))
##					t.cleanup()
## 
##				# update the last uploaded timestamp and reset the motion
##				# counter
##				lastUploaded = timestamp
##				motionCounter = 0
## 
##	# otherwise, the room is not occupied
##	else:
##		motionCounter = 0
        
        if conf["detect_faces"]:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]
                                
	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the security feed
		#cv2.imshow("Security Feed", frame)
		cv2.imwrite("output.jpg", frame)
		key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break

                
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
