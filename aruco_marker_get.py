import cv2
import numpy as np
from ArucoMarker import ArucoMarker
from ArucoMarkerInfo import ArucoMarkerInfo
from Quadrilaterals import Quadrilateral
from corner import CornerRefinement
from math import fabs, sqrt
import time

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
    
def process_aruco_image(image):
    aruco = cv2.resize(image, (8, 8))
    gray = cv2.cvtColor(aruco, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
    
    
def read_aruco_data(binary):
    marker = ArucoMarker()
    cells = binary.flatten()
    marker.cells = cells.reshape((8, 8)).astype(np.int32)
    return marker

def deform_quad(image, size, quad):
    points = np.array([
        [0, 0],
        [0, size[1]],
        [size[0], size[1]],
        [size[0], 0]
    ], dtype=np.float32)
    
    quad = np.array(quad, dtype=np.float32)
    transformation = cv2.getPerspectiveTransform(quad, points)
    out = cv2.warpPerspective(image, transformation, size, flags=cv2.INTER_LINEAR)
    cv2.imwrite('yo.jpg', out)    

    return out


# Load image and convert to binary
image = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('images.jpeg')
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
blurred = cv2.GaussianBlur(image, (3, 3), 0)
edged = cv2.Canny(blurred, 10, 100)
# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
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
        #print (approx)
        for point in approx:
            # Extract the x and y coordinates of the point
            corner = point[0]
            cor= CornerRefinement.refine_corner_harris(image1,corner,10)
            point[0]= cor
            #cv2.circle(image1, cor, 1, (0, 255, 0), -1) 
            #cv2.circle(image1, corner, 1, (0, 0, 255), -1) 
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

#draw_quads(image1, squares)
markers = []
start_time= time.time()


quad=squares[12]
board = deform_quad(image1, (64, 64), quad.points)
#pers= cv2.imread('perspective.png')
binary = process_aruco_image(board)
marker = read_aruco_data(binary)
end_time= time.time()
print(end_time-start_time)
marker.projected = quad.points
bad=0
print(marker.cells)
marker.cells = (marker.cells >= 127).astype(int)
print (marker.cells)
#print (marker.print())

for i in range (4):
    print ('_____rotation_______', i+1)
    #print(marker.hamming_distance())
    print (marker.calculate_id())
    marker.rotate()
    
    print (marker.cells)
    #print(marker.hamming_distance())

#if marker.validate():
        #markers.append(marker)
#print (squares)
#print (marker.print())
# Show the image with the drawn squares'''
cv2.imwrite('squares_output_get_marker.jpg', image1)