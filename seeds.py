import cv2 as cv
import numpy as np
import sys
import random




#read image
img = cv.imread('images-kaggle/image-30.png') 
converted_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)


height,width,channels = converted_img.shape
num_iterations = 10
prior = 2
double_step = False
num_superpixels = 200
num_levels = 4
num_histogram_bins = 5


seeds = cv.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
color_img = np.zeros((height,width,3), np.uint8)
color_img[:] = (0, 0, 255)
seeds.iterate(converted_img, num_iterations)


# retrieve the segmentation result
labels = seeds.getLabels()


# labels output: use the last x bits to determine the color
num_label_bits = 2
labels &= (1<<num_label_bits)-1
labels *= 1<<(16-num_label_bits)


mask = seeds.getLabelContourMask(False)
obtainedSuperpixels = seeds.getNumberOfSuperpixels()
meanColors = []




# stitch foreground & background together
mask_inv = cv.bitwise_not(mask)
result_bg = cv.bitwise_and(img, img, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=mask)
result = cv.add(result_bg, result_fg)


cv.namedWindow('mask',0)
cv.namedWindow('result_bg',0)
cv.namedWindow('result_fg',0)
cv.namedWindow('result',0)


cv.imshow('mask',mask_inv)
cv.imshow('result_bg',result_bg)
cv.imshow('result_fg',result_fg)
cv.imshow('result',result)



cv.waitKey(0)