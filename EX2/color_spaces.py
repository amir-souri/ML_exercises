# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import numpy as np
import cv2

filename = './inputs/lena.jpg'

# <Exercise 2.2>

img = cv2.imread(filename)

imrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imhsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("RGB", imrgb)
cv2.imshow("HSV", imhsv)

# [x,y,color]

r = imrgb[:, :, 0]
g = imrgb[:, :, 1]
b = imrgb[:, :, 2]

cv2.imshow("r", r)
cv2.imshow("g", g)
cv2.imshow("b", b)

r1, g1, b1 = cv2.split(imrgb)

cv2.imshow("r1", r1)
cv2.imshow("g1", g1)
cv2.imshow("b1", b1)

# 5.Change the channel representation:


# print(r1.shape)

zero = np.zeros(r1.shape, dtype=np.uint8)

# You can either use () or []
channelG1 = cv2.merge([zero, g1, zero])
channelR1 = cv2.merge((zero, zero, r1))
channelB1 = cv2.merge([b1, zero, zero])

channelG = cv2.merge((zero, g, zero))
channelR = cv2.merge([zero, zero, r])
channelB = cv2.merge([b, zero, zero])

cv2.imshow("Green1", channelG1)
cv2.imshow("Red1", channelR1)
cv2.imshow("Blue1", channelB1)

cv2.imshow("Green", channelG)
cv2.imshow("Blue", channelB)
cv2.imshow("Red", channelR)

cv2.waitKey(0)
