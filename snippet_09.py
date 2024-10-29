# snippet_09.py
# -----------------------

"""
Python Code Snippets
Terence Morley, Department of Computer Science,
    The University of Manchester, Nov 2020

Purpose: Threshold a greyscale image.
"""

import cv2
import sys
from matplotlib import pyplot as plt
import numpy as np


input_file = 'images.jpeg'
img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
#
# Check for success
if img is None:
    print('Failed to open', input_file)
    sys.exit()


# Threshold manually at intensity level 150
# Note that threshold() returns the computued threshold value
# and the resulting image.  We don't need the value so we put
# it in _.
_, output = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
#cv2.imwrite('thresh_150.jpg', output)
contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imwrite('thresh_150.jpg', img)

binary_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 5)

mean_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 11, 5)

cv2.imwrite('thresh_adaptive_binary.jpg', binary_image)
cv2.imwrite('thresh_adaptive_mean.jpg', mean_img)
# Threshold automatically with Otsu's method
# This time, we do want the computed threshold value
T, output = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite('thresh_otsu.jpg', output)
print("Threshold value found by Otsu's method:", T)
