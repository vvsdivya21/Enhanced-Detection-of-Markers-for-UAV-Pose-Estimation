import cv2
import numpy as np
import os
from corner import CornerRefinement
import time

# Load image and convert to binary
start_time_1 = time.time()

image = cv2.imread(os.path.join('/home/mscrobotics2324laptop9/Downloads','WhatsApp Image 2024-08-27 at 21.31.59.jpeg'), cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread(os.path.join('/home/mscrobotics2324laptop9/Downloads','WhatsApp Image 2024-08-27 at 21.31.59.jpeg'))
im=cv2.imread(os.path.join('/home/mscrobotics2324laptop9/Downloads','1.jpeg'))
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#------CANNY
blurred = cv2.GaussianBlur(image, (3, 3), 0)
edged = cv2.Canny(blurred, 10, 100)#-----CANNY
kernel = np.ones((5, 5), np.uint8)  # Increase the size for thicker edges

dilated_edges = cv2.dilate(edged, kernel, iterations=1)
#cv2.imwrite(os.path.join('/home/mscrobotics2324laptop9/images_dissertation','canny_output.jpg'), dilated_edges)   

#binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)#-------ADAPTIVE

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #------CANNY
#contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #------ADAPTIVE

color_img1= cv2.cvtColor(binary_image,cv2.COLOR_GRAY2BGR)
color_img2= cv2.cvtColor(binary_image,cv2.COLOR_GRAY2BGR)
# Process each contour
c=0
a=0
corner_coords=[]


areas=[]

error_list=[]
c=0
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    #print (approx)
    # Draw the contour and the approximated polygon
    #print ('c',c, contour)
    c = contour[0]
    a=0
    # Draw the original contour
    

    if len(approx)==4 and cv2.isContourConvex(approx) and cv2.contourArea(approx)>7000:
        c+=1
        print (approx)
        cv2.drawContours(image1, [approx], -1, (0, 255, 0), 2)
        for point in approx:
            # Extract the x and y coordinates of the point
            corner = point[0]
            x,y= corner
            #cv2.putText(color_img1, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            cor= CornerRefinement.refine_corner_harris(image1,corner,10)
            point= cor

            areas.append(cv2.contourArea(approx))
            print (corner, cor, point,approx)
            
            #center = (int(centroids[0]), int(centroids[1]))
            m,n= cor
            #cv2.circle(color_img1, corner, 3, (0, 0, 255), -1) #red-prev
            #cv2.circle(color_img1, cor, 10, (0, 255, 0), -1) #green-harris
            #cv2.putText(color_img1, f'({m},{n})', (m, n - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            
            error = np.linalg.norm(np.array(corner) - np.array(cor))
            a+=error
            print(f'Error between original and refined corner: {error}')
        error_list.append(a)
        print(c, error_list)
            #cv2.imwrite('area{}.png'.format(point),dst)
#lower_bound = np.percentile(areas,25 )
#upper_bound = np.percentile(areas, 99)
#outliers = [x for x in areas if x < lower_bound ]


print (error_list)
print (areas)
end_time_1 = time.time()
elapsed_time_1 = end_time_1 - start_time_1
print(elapsed_time_1)
cv2.imwrite(os.path.join('/home/mscrobotics2324laptop9/images_dissertation','squares.jpg'), image1)   



            # Draw a small circle at each contour point
            #cv2.circle(color_img1, center, 2, (0, 255, 0), -1) 
            #cv2.putText(color_img1, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # Print the vertices of the approximated polygon
    #print ('a',a,approx)
    
# Save and show the result
#color_img1[centroids[:,1],centroids[:,0]]=[0,255,0]
'''cv2.imwrite('contours_approx_output.jpg', color_img1)
cv2.imwrite('contours_approx_output2.jpg', color_img2)
'''
