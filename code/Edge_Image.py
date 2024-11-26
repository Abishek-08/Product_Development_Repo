import cv2 as cv
import os
import numpy as np

#Reading the image
rawImage = cv.imread(os.path.join('.','data','basketball.jpg'))

#Edge Detection using canny method
image_canny = cv.Canny(rawImage,100,400)

#using Dilate method -> make image to more thicker
image_Dilate = cv.dilate(image_canny,np.ones((4,4),dtype=int))

#using erode method -> make image to more thinner (opposite to the Dilate method)
image_erode = cv.erode(image_Dilate,np.ones((4,4),dtype=int))

#visualize the Image
cv.imshow('Raw-Image',rawImage)
cv.imshow('Canny-Image',image_canny)
cv.imshow('Dilate-Image',image_Dilate)
cv.imshow('Erode-Image',image_erode)
cv.waitKey(0)