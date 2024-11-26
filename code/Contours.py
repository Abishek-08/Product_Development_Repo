import cv2 as cv
import os

#Read the Image
rawImage = cv.imread(os.path.join('.','data','flyingbirds.jpg'))

#Convert image into gray scale
gray_Image = cv.cvtColor(rawImage,cv.COLOR_BGR2GRAY)

#Applying Threshold
ret,thresh = cv.threshold(gray_Image,100,300,cv.THRESH_BINARY_INV) 

#Applying the Contours -> it will return the group of separated object which were stored in the contours like list
contours,hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    print(cv.contourArea(cnt))
    if cv.contourArea(cnt)>500:
        #cv.drawContours(rawImage,cnt,-1,(0,0,255),2) #It will draw the edge around the object ->syntax (image,contoruslist,contoursIntValue,color,thickness)

        x1,y1,w,h = cv.boundingRect(cnt) #It will return the (x1,y1) points as well as height and width also using this point, we manually the draw the rectangle for the each object detect in the image

        cv.rectangle(rawImage,(x1,y1),(x1+w,y1+h),(0,0,255),2)

#visualize the Image
cv.imshow('Raw-Image',rawImage)
cv.imshow('Gray-Image',gray_Image)
cv.imshow('Threshold-Image',thresh)
cv.waitKey(0)