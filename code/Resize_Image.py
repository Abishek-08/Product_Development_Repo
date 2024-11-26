import cv2 as cv
import os

#Raw_Image without Resizing
raw_Image_path = os.path.join('.','data','bird_01.jpg')
raw_Image = cv.imread(raw_Image_path)
print(raw_Image.shape)

#After Resizing the Image, Here measurement formart is (width, height)
resized_Image = cv.resize(raw_Image, (600,400))
print(resized_Image.shape)

cv.imshow('RawImage',raw_Image)
cv.imshow('ResizedImage',resized_Image)
cv.waitKey(0)