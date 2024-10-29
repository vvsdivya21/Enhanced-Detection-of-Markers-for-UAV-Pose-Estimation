import cv2
import numpy as np
from corner import CornerRefinement
from math import fabs, sqrt
from Quadrilaterals import Quadrilateral


limit_cosine=0.6
min_area=0 
max_error=0.02
def draw_quads(mat, quads):
        for quad in quads:
            for j in range(4):
                cv2.line(mat, tuple(quad.points[j]), tuple(quad.points[(j + 1) % 4]), (255, 0, 255), 2)

def angle_corner_points_cos(b, c, a):
        dx1 = b[0] - a[0]
        dy1 = b[1] - a[1]
        dx2 = c[0] - a[0]
        dy2 = c[1] - a[1]
        dot_product = dx1 * dx2 + dy1 * dy2
        magnitude1 = dx1 * dx1 + dy1 * dy1
        magnitude2 = dx2 * dx2 + dy2 * dy2
        epsilon = 1e-12  # Increased epsilon to avoid precision issues
        denom = sqrt(max(magnitude1 * magnitude2, epsilon))
        
        if denom == 0:
            print(f"Zero denominator encountered with points a={a}, b={b}, c={c}")
            return 0  # Return 0 cosine if denominator is zero to avoid division by zero
        
        return dot_product / denom
# Load image and convert to binary
image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('images.jpeg')
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
color_img1= cv2.cvtColor(binary_image,cv2.COLOR_GRAY2BGR)
color_img2= cv2.cvtColor(binary_image,cv2.COLOR_GRAY2BGR)
# Process each contour
areas=[]

squares=[]
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    
    if len(approx)==4 and cv2.isContourConvex(approx):
        max_cosine = 0

        for point in approx:
            # Extract the x and y coordinates of the point
            corner = point[0]
            cor= CornerRefinement.refine_corner_harris(image1,corner,10)
            point[0]= cor

        for j in range(2, 5):
            cosine = fabs(angle_corner_points_cos(approx[j % 4][0], approx[j - 2][0], approx[j - 1][0]))
            max_cosine = max(max_cosine, cosine)

            if max_cosine < limit_cosine:
                quad = Quadrilateral(
                    approx[0][0],
                    approx[1][0],
                    approx[2][0],
                    approx[3][0]
                    )
                 
                squares.append(quad)

draw_quads(color_img1, squares)

    # Show the image with the drawn squares
cv2.imwrite('squares_output.jpg', color_img1)

#cv2.imwrite('contours_approx_output.jpg', color_img1)        
'''
            # Draw a small circle at each contour point
            #cv2.circle(color_img1, center, 2, (0, 255, 0), -1) 
            #cv2.putText(color_img1, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # Print the vertices of the approximated polygon
    #print ('a',a,approx)
    a+=1
# Save and show the result
#color_img1[centroids[:,1],centroids[:,0]]=[0,255,0]
cv2.imwrite('contours_approx_output.jpg', color_img1)
cv2.imwrite('contours_approx_output2.jpg', color_img2)'''

