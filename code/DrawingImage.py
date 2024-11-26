import cv2 as cv
import os

#Reading the Image
rawImage = cv.imread(os.path.join('.','data','board.jpg'))
print(rawImage.shape)

#Line (image,(point1),(point2),(BGR-color),thickness)
cv.line(rawImage,(40,100),(100,300),(255,0,0),4)

#Rectangle (image,(point1),(point2),(BGR-color),thickness) (negative value of thickness means shape the color of the shape)
cv.rectangle(rawImage,(20,60),(100,150),(0,255,0),-2)

#Circle (image,(center-point),radius,(color),thickness)
cv.circle(rawImage,(160,80),70,(0,0,255),4)

#Text (image,stringmsg,(originpoint),fontStyle,fontsize,(color),thickness)
cv.putText(rawImage,'Abishek',(100,215),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),4)

#visualize the Image
cv.imshow('Raw-Image',rawImage)
cv.waitKey(0)