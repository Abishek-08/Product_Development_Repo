import cv2 as cv
import os

#Reading the RawImage, before Cropping
raw_Image = cv.imread(os.path.join('.','data','bird_01.jpg'))
print(raw_Image.shape)

#Crop the RawImage, After Cropping, syntax image[rowStart:rowEnd,colStart:colEnd],[height,width]
cropped_Image = raw_Image[100:200,200:280]
print(cropped_Image.shape)

#Visualize the Image
cv.imshow('RawImage',raw_Image)
cv.imshow('croppedImage',cropped_Image)
cv.waitKey(0)
