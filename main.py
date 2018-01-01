# -*- coding: utf-8 -*-
import sys
from imutils.perspective import four_point_transform
import numpy as np
import cv2

im = cv2.imread("calc1.jpg")
iminput = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(im,(5,5),0)
edged = cv2.Canny(blur, 50, 200, 255)

im2 ,contours, hierarchy = cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#c = max(contours, key = cv2.contourArea)

cnts = sorted(contours,key = cv2.contourArea, reverse=True)
displayCnt = None
flag = 0
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # if the contour has four vertices, then we have found
    #the calculator
    if len(approx) == 4:
        displayCnt = approx
        break

warped = four_point_transform(im, displayCnt.reshape(4, 2))
output = four_point_transform(im, displayCnt.reshape(4, 2))

cv2.imshow("display",warped)
blur1 = cv2.GaussianBlur(warped,(5,5),0)
edged1 = cv2.Canny(blur1, 50, 200, 255)

im3 ,contoursn, hierarchy = cv2.findContours(edged1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)
for c in contoursn:
    x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
    cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),1)

c = max(contoursn, key = cv2.contourArea)
x,y,w,h = cv2.boundingRect(c)
    # draw the book contour (in green)
cv2.rectangle(warped,(x,y),(x+w,y+h),(0,0,255),1)
cv2.imshow("display?",warped)
'''rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(im,[box],0,(0,0,255),2)
'''
#cv2.drawContours(warped, contoursn, -1, (0,255,0), 3)
cv2.imshow("img",im)
cv2.imshow("gray",edged1)
