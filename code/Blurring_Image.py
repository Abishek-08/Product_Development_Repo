import cv2 as cv
import os

#Bluring concept is used to remove the noise from the noisy images, it is the primary use-case

#Read the Image
raw_Image = cv.imread(os.path.join('.','data','person_01.jpg'))
noisy_Image = cv.imread(os.path.join('.','data','noisyImage_01.png'))
k_size=7

#classical Blur
blur_Image = cv.blur(raw_Image,(k_size,k_size))
blur_Noisy_Image = cv.blur(noisy_Image,(k_size,k_size))

#Gaussian Blur
gaussian_Blur_Image = cv.GaussianBlur(raw_Image,(k_size,k_size),5)
gaussian_Noisy_Image = cv.GaussianBlur(noisy_Image,(k_size,k_size),4)

#Median Blur
median_Blur_Image = cv.medianBlur(raw_Image,k_size)
median_Noisy_Image = cv.medianBlur(noisy_Image,k_size)

#Visualize the Image
cv.imshow('Raw-Image',raw_Image)
cv.imshow('ClassicBlur',blur_Image)
cv.imshow('GaussianBlur',gaussian_Blur_Image)
cv.imshow('MedianBlur',median_Blur_Image)

cv.imshow('Raw-Noisy', noisy_Image)
cv.imshow('classicBlur',blur_Noisy_Image)
cv.imshow('gaussianBlur',gaussian_Noisy_Image)
cv.imshow('medain-noisy',median_Noisy_Image)
cv.waitKey(0)