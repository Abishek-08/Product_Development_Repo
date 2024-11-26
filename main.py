import cv2 as cv
import os
from PIL import Image
from util import get_Color_Limits 

#Give the Input color to Find
inputColor = [0, 255, 255]

#Read the WebCam
webCam = cv.VideoCapture(0)

#Visualize the Webcam Video
while True:
    ret, frames = webCam.read()

    hsvFrameImage = cv.cvtColor(frames,cv.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_Color_Limits(color=inputColor)

    #InRange Function mask the image in the particular range of the object
    maskFrameImage = cv.inRange(hsvFrameImage,lowerLimit,upperLimit)

    #getting the masked image as an array
    maskFrameImage_Array = Image.fromarray(maskFrameImage)

    #from the maskedArray get boundBox to detect or draw the bounding box around the object. It will two points point1(x1,y1) and points2(x2,y2) these are the dialognal point for the rectangle around the object
    boundingBox = maskFrameImage_Array.getbbox()
    print(boundingBox)

    #From bounding points draw the rectangle around the objects
    if boundingBox is not None:
        x1,y1,x2,y2 = boundingBox
        frames = cv.rectangle(frames,(x1,y1),(x2,y2),(0,0,255),4)

    cv.imshow('WebCam',frames)
    cv.imshow('mask-Cam',maskFrameImage)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

webCam.release()
cv.destroyAllWindows()