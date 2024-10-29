import cv2
import numpy as np
import threading
from corner import CornerRefinement
from math import fabs, sqrt
from Quadrilaterals import Quadrilateral
import matplotlib.pyplot as plt
import time



limit_cosine=0.6
min_area=0 
max_error=0.02
markerWarpPixSize=5
def get_area(corners):
    # use the cross products
    v01 = (corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    v03 = (corners[3][0] - corners[0][0], corners[3][1] - corners[0][1])
    area1 = abs(v01[0] * v03[1] - v01[1] * v03[0])

    v21 = (corners[1][0] - corners[2][0], corners[1][1] - corners[2][1])
    v23 = (corners[3][0] - corners[2][0], corners[3][1] - corners[2][1])
    area2 = abs(v21[0] * v23[1] - v21[1] * v23[0])

    return (area2 + area1) / 2.0

def draw_quads(mat, quads):
        for quad in quads:
            for j in range(4):
                cv2.line(mat, tuple(quad.points[j]), tuple(quad.points[(j + 1) % 4]), (255, 0, 255), 2)
        cv2.imwrite('low_res_aruco.png', mat )     

def warp(in_img, size, points):
    if len(points) != 4:
        raise ValueError("points.size() != 4")

    # Obtain the perspective transform
    points_in = np.array(points, dtype=np.float32)
    points_res = np.array([
        [0, 0],
        [size[0] - 1, 0],
        [size[0] - 1, size[1] - 1],
        [0, size[1] - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(points_in, points_res)
    
    # Prepare the output image with the desired size
    out_img = cv2.warpPerspective(in_img, M, (size[0], size[1]), flags=cv2.INTER_LINEAR)
    
    return out_img

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
def get_min_marker_size_pix(orginput_image_size):
    """
    Calculate the minimum marker size in pixels.

    :param orginput_image_size: Tuple representing the size of the original input image (width, height)
    :return: Minimum marker size in pixels
    """
    min_size = 0.5
    min_size_pix = -1

    if min_size == -1 and min_size_pix == -1:
        return 0

    # Calculate the maximum dimension of the input image
    max_dim = max(orginput_image_size[0], orginput_image_size[1])
    size = 0

    if min_size != -1:
        size = min_size * max_dim
    
    if min_size_pix != -1:
        size = min(min_size_pix, size)
    
    return size
def get_marker_warp_size():
    bis = -1
    if bis != -1:
        return bis

    ndiv = -1
    if ndiv == -1:
        ndiv = 7  # set any possible value (it is used for non-dictionary based labelers)
    
    return markerWarpPixSize * ndiv  # this is the minimum size that the smallest marker will have

def build_pyramid(image_pyramid, grey, min_size, pyrfactor):
    # Determine the number of pyramid images
    npyrimg = 1
    imgpsize = grey.shape[1], grey.shape[0]  # (width, height)
    
    while imgpsize[0] > min_size:  # imgpsize[0] is the width
        imgpsize = (imgpsize[0] // pyrfactor, imgpsize[1] // pyrfactor)
        npyrimg += 1

    # Resize the image pyramid to the required number of images
    image_pyramid.extend([None] * (npyrimg - len(image_pyramid)))
    image_pyramid[0] = grey

    # Now, create pyramid images
    for i in range(1, npyrimg):
        nsize = (image_pyramid[i - 1].shape[1] // pyrfactor, 
                 image_pyramid[i - 1].shape[0] // pyrfactor)
        image_pyramid[i] = cv2.resize(image_pyramid[i - 1], nsize)

input_img= cv2.imread('imgdiss.jpeg')
# Convert to grayscale if necessary
if len(input_img.shape) == 3 and input_img.shape[2] == 3:
    grey = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
else:
    grey = input_img

#--------------------------------------------'LOW RES'------------------------------------------------
lowResMarkerSize= 20
# Create low-resolution image if needed
resize_factor = 0.4 #1.0
max_image_size = grey.shape[::-1]
minpixsize = get_min_marker_size_pix(input_img.shape)
if lowResMarkerSize < minpixsize:
    resize_factor = lowResMarkerSize / minpixsize
    if resize_factor < 0.9:
        max_image_size = (int(grey.shape[1] * resize_factor + 0.5), int(grey.shape[0] * resize_factor + 0.5))
        max_image_size = (max_image_size[0] + (max_image_size[0] % 2), max_image_size[1] + (max_image_size[1] % 2))
        img_to_be_thresholded = cv2.resize(grey, max_image_size, interpolation=cv2.INTER_NEAREST)
    else:
        img_to_be_thresholded = grey
else:
    img_to_be_thresholded = grey
#--------------------------------------------'PYRAMID'------------------------------------------------
# Build image pyramid if needed
pyrfactor = 2
need_pyramid = resize_factor < 1 / pyrfactor
print (need_pyramid)
image_pyramid = [grey]
if need_pyramid:
    build_pyramid_thread = threading.Thread(target= build_pyramid, args=(image_pyramid, grey,2,pyrfactor))
    build_pyramid_thread.start()
    build_pyramid_thread.join()
else:
    image_pyramid = [grey]

# Display the pyramid
max_width = max(img.shape[1] for img in image_pyramid)
padded_pyramid = [cv2.copyMakeBorder(img, 0, 0, 0, max_width - img.shape[1], cv2.BORDER_CONSTANT, value=0) for img in image_pyramid]

# Stack the images vertically
stacked_image = np.vstack(padded_pyramid)

#im=np.vstack(image_pyramid)
cv2.imwrite('low_res.png', stacked_image)
#print (i,img)
print ('executed till end')


#--------------------------------------------'SQUARES DETECT'------------------------------------------------

image_gray = cv2.imread('imgdiss.jpeg', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

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
            #cor= CornerRefinement.refine_corner_harris(input_img,corner,10)
            #point[0]= cor

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
                #print (quad)

                squares.append(quad)
draw_quads(input_img, squares)

#--------------------------------------------'warp'------------------------------------------------
start_time_1 = time.time()
marker_warp_size= get_marker_warp_size()
if need_pyramid:
    build_pyramid_thread.join()

# Candidate classification
hist = [0] * 256
'''
for i, candidate in enumerate(squares):
    print (i,candidate)
    in_to_warp = img_to_be_thresholded

    if need_pyramid:
        img_pyr_idx = 0
        print ('tru')

for quad in squares:
            for j in range(4):
'''
for i, candidate in enumerate(squares):
    in_to_warp = img_to_be_thresholded
    if need_pyramid:
        img_pyr_idx = 0
        for p in range(1, len(image_pyramid)):
            #print(get_area(candidate.points))
            if get_area(candidate.points) / (4 ** p) >= get_marker_warp_size() ** 2:
                print ('truuuuu')
                img_pyr_idx = p
            else:
                break
        in_to_warp = image_pyramid[img_pyr_idx]
        ratio = in_to_warp.shape[1] / img_to_be_thresholded.shape[1]
        for j in range(len(candidate.points)):
            candidate.points[j] = (candidate.points[j] * ratio).astype(np.float32)
        # Adjust points to the image level p
        #candidate.scale_points(ratio)

    canonical_marker = warp(in_to_warp, (marker_warp_size,marker_warp_size),candidate.points)
#cv2.imwrite('low_res.png',canonical_marker)
end_time_1 = time.time()
elapsed_time_1 = end_time_1 - start_time_1
print (elapsed_time_1)