import cv2
import numpy
import os

import cv2.version

print(cv2.__version__ )
print(numpy.__version__ )

#Read the Image
image_path=os.path.join('./Python Pratice','data','bird_01.jpg')

image = cv2.imread(image_path)

#write the Image
cv2.imwrite(os.path.join('.','data','bird_01_out.jpg'),image)

#Visualize and Image
cv2.imshow('Image', image)
cv2.waitKey(0)
