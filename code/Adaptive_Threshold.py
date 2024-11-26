import cv2 as cv
import os

#Read the Image
rawImage = cv.imread(os.path.join('.','data','handwritten.jpg'))

#Resize the Image
resized_Image = cv.resize(rawImage,(650,500))

#Convert the image into grayScale
gray_Image = cv.cvtColor(resized_Image,cv.COLOR_BGR2GRAY)

#Apply the normal Threshold
ret,global_thresh = cv.threshold(gray_Image,80,600,cv.THRESH_BINARY)

#Applying the AdaptiveThreshold to the Image
thresh = cv.adaptiveThreshold(gray_Image,100,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,23,60)

#Visualize the Image
cv.imshow('RawImag',resized_Image)
cv.imshow('Adaptive-Threshold-Image',thresh)
cv.imshow('GlobalThreshold',global_thresh)
cv.waitKey(0)