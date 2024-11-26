import cv2 as cv
import os

#Reading the video Clip
video_path = os.path.join('.','data','CarRacing.mp4')
video = cv.VideoCapture(video_path)

#Visualize the video Clip
ret = True

while ret:
    ret,frames = video.read()

    if ret:
        cv.imshow('carVideo',frames)
        cv.waitKey(100)

video.release()
cv.destroyAllWindows()

