import cv2 as cv
import os

#Reading from the WebCam
webCam = cv.VideoCapture(1)

while True:
    ret, frames = webCam.read()
    frames = cv.resize(frames,(860,480))
    cv.imshow('liveCam', frames)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

webCam.release()
cv.destroyAllWindows
 