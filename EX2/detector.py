# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import argparse
import cv2
import numpy as np

# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the (optional) video file")
args = vars(ap.parse_args())

# Load a video camera or a video file.
if not args.get("video", False):
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture(args["video"])

# Grab each individual frame.
while True:
    # Grabs, decodes and returns the next video frame.
    retval, frame = video.read()

    # Check if there is a valid frame.
    if not retval:
        break

    # <Exercise 2.6>

    # Remove this line after you finish the exercise.
    # result = frame.copy()

    # <Exercise 2.4 (a)>
    imrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # <Exercise 2.4 (b)>
    # define range of blue color in HSV
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([255, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(imhsv, lower_blue, upper_blue)

    # <Exercise 2.4 (c)>
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the processed images.
    cv2.imshow("Result", frame)

    # <Exercise 2.4 (e)>
    # print(imhsv.shape) = (480, 640, 3)  number of rows, columns, and channels


    # Get the keyboard event.j
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Closes video file or capturing device.
video.release()


cv2.imshow("HSV", imhsv)
cv2.imshow("Mask", mask)
cv2.imshow("Final", res)

##########
emptyMask = np.zeros([480, 640], dtype="uint8")
inde = (imhsv > lower_blue ) & (imhsv < upper_blue)
inde = np.all(inde, axis=2)
emptyMask[inde] = 255
res = frame & emptyMask.reshape((*emptyMask.shape , 1))
cv2.imshow("Amir", res)
##############

cv2.waitKey(0)

# Destroys all of the HighGUI windows.
cv2.destroyAllWindows()