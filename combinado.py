#Programacion que permite revisar los frames que esta
#arrojando la camara corriendo una programacion
import cv2
import time
import numpy as np

#Numero 0 para activar la webcam
#Numero 1 para activar la camara de USB
video = cv2.VideoCapture(0)

while True:
	    # Find OpenCV version
	    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

	    # Number of frames to capture
	    num_frames = 1;
	    print ("Capturing {0} frames".format(num_frames))

	    # Start time
	    start = time.time()
	    # Grab a few frames

	    for i in range(0, num_frames):
	        ret, frame = video.read()
	        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	        mask = cv2.inRange(hsv,np.array([64,49,75]),np.array([81,255,255]))
	        median = cv2.medianBlur(mask,15)
	        cv2.imshow('VideoOutput', median)

	    # End time
	    end = time.time()

	    # Time elapsed
	    seconds = end - start
	    print ("Time taken : {0} seconds".format(seconds))

	    # Calculate frames per second
	    fps  = num_frames / seconds;
	    print ("Estimated frames per second : {0}".format(fps))
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
# Release video
video.release()