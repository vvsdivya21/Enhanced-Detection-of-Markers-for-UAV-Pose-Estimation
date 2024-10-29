import cv2
import numpy as np
from ArucoMarker import ArucoMarker
from ArucoMarkerInfo import ArucoMarkerInfo
from Quadrilaterals import Quadrilateral
from corner import CornerRefinement
from math import fabs, sqrt
import os, time
start_time_1 = time.time()
contour_try= [[[258,  23]],

 [[257,  34]],

 [[270,  36]],

 [[271,  25]]]

class MarkerCandidate:
    def __init__(self, contour=None, idx=None):
        self.contour = contour if contour is not None else []
        self.idx = idx
def find_corner_points_in_contour(points, contour):
    """
    Finds the indices of four corner points in the given contour.

    :param points: List of 4 points (cv2.Point2f) representing the corners.
    :param contour: List of contour points (cv2.Point).
    :return: A list of indices in the contour corresponding to the corner points.
    """
    assert len(points) == 4
    idx_segments = [-1, -1, -1, -1]
    #points2i = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
    points2i = np.array([tuple(map(int, p.flatten())) for p in points], dtype=np.int32)
    #print (points2i)
    contour = np.array([tuple(map(int, pt.flatten())) for pt in contour], dtype=np.int32)
    #print (contour)   

    for i, pt in enumerate(contour):
        if idx_segments[0] == -1 and (pt[0], pt[1]) == tuple(points2i[0]):
            idx_segments[0] = i
        if idx_segments[1] == -1 and (pt[0], pt[1]) == tuple(points2i[1]):
            idx_segments[1] = i
        if idx_segments[2] == -1 and (pt[0], pt[1]) == tuple(points2i[2]):
            idx_segments[2] = i
        if idx_segments[3] == -1 and (pt[0], pt[1]) == tuple(points2i[3]):
            idx_segments[3] = i
    
    return idx_segments

def find_deformed_sides_idx(contour, idx_segments):
    """
    Determines which sides of the contour are deformed due to cylinder perspective.

    :param contour: List of contour points (cv2.Point).
    :param idx_segments: Indices of the corner points in the contour.
    :return: The index of the deformed side (0 or 1).
    """
    dist_sum = [0, 0, 0, 0]
    
    def compute_distance(p1, p2, contour, start_idx, end_idx):
        """Computes the average perpendicular distance of contour points from the line segment (p1, p2)."""
        inv_den = 1.0 / np.linalg.norm(np.array(p2) - np.array(p1))
        dist = 0
        for j in range(start_idx, end_idx):
            point = np.array(contour[j])
            line_vector = np.array(p2) - np.array(p1)
            point_vector = np.array(point) - np.array(p1)
            dist += abs(np.cross(line_vector, point_vector)) * inv_den
        #print (dist / (end_idx - start_idx) if end_idx != start_idx else 0)
        return dist / (end_idx - start_idx) if end_idx != start_idx else 0
        
    for i in range(3):
        p1 = contour[idx_segments[i]]
        p2 = contour[idx_segments[i + 1]]
        dist_sum[i] = compute_distance(p1, p2, contour, idx_segments[i], idx_segments[i + 1])

    # For the last segment
    p1 = contour[idx_segments[0]]
    p2 = contour[idx_segments[3]]
    dist_sum[3] = compute_distance(p1, p2, contour, 0, idx_segments[0]) + compute_distance(p1, p2, contour, idx_segments[3], len(contour))
    dist_sum[3] /= (idx_segments[0] + (len(contour) - idx_segments[3]))
    print (dist_sum)
    if dist_sum[0] + dist_sum[2] > dist_sum[1] + dist_sum[3]:
        return 0
    else:
        return 1

def warp_cylinder(in_image, size, pts, ctr):
    """
    Warps the given image to correct for cylinder perspective distortion.

    :param in_image: Input image to be warped (cv2.Mat).
    :param size: Size of the output image (width, height).
    :param mcand: MarkerCandidate object containing contour and points.
    :return: True if successful, otherwise False.
    """
    #assert len(mcand) == 4

    idx_segments = find_corner_points_in_contour(pts, ctr)
    print (idx_segments)
    min_idx = idx_segments.index(min(idx_segments))
    idx_segments = idx_segments[min_idx:] + idx_segments[:min_idx]
    #mcand = mcand[min_idx:] + mcand[:min_idx]
    pts = np.concatenate((pts[min_idx:], pts[:min_idx]))


    defrmd_side = find_deformed_sides_idx(ctr, idx_segments)
    print (defrmd_side)
    #center = np.mean(mcand, axis=0)
    enlarged_region = np.array(pts, dtype=np.float32)
    if defrmd_side == 0:
        enlarged_region[0] += (pts[3] - pts[0]) * 1.2
        enlarged_region[1] += (pts[2] - pts[1]) * 1.2
        enlarged_region[2] += (pts[1] - pts[2]) * 1.2
        enlarged_region[3] += (pts[0] - pts[3]) * 1.2
    else:
        enlarged_region[0] += (pts[1] - pts[0]) * 1.2
        enlarged_region[1] += (pts[0] - pts[1]) * 1.2
        enlarged_region[2] += (pts[3] - pts[2]) * 1.2
        enlarged_region[3] += (pts[2] - pts[3]) * 1.2

    enlarged_size = (int(size[0] * 1.2), int(size[1] * 1.2))
    points_res = np.array([[0, 0], [enlarged_size[0] - 1, 0], [enlarged_size[0] - 1, enlarged_size[1] - 1], [0, enlarged_size[1] - 1]], dtype=np.float32)

    if defrmd_side == 0:
        points_res = np.roll(points_res, shift=1, axis=0)

    M = cv2.getPerspectiveTransform(enlarged_region, points_res)
    im_aux = cv2.warpPerspective(in_image, M, enlarged_size, flags=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join('/home/mscrobotics2324laptop9/images_dissertation','deformed_warped1.png'), im_aux)
    points_co = [cv2.perspectiveTransform(np.array([pt], dtype=np.float32), M)[0][0] for pt in pts]
    #print (points_co)
    im_aux2 = np.zeros(enlarged_size, dtype=np.uint8)
   
    for point in points_co:
        x, y = np.int32(point)
        im_aux2[y, x] = 255
        if y > 0:
            im_aux2[y - 1, x] = 255
        if y < im_aux2.shape[0] - 1:
            im_aux2[y + 1, x] = 255
    
    out_im = np.zeros(enlarged_size, dtype=np.uint8)
    for y in range(im_aux2.shape[0]):
        start, end = -1, -1
        for x in range(im_aux2.shape[1]):
            if im_aux2[y, x]:
                if start == -1:
                    start = x
                else:
                    end = x
        if start != -1 and end != -1: # and (end - start) > size[0] >> 1:
            end = min(end + 1, im_aux.shape[1])
            #out_im[y, :] = im_aux[y, start:end + 1]
            out_im[y, :end - start] = im_aux[y, start:end]
    
    out = out_im[:, :size[0]]
    cv2.imwrite ('out_im.png', out)
    return True, out_im

# Load image and convert to binary
image = cv2.imread(os.path.join('/home/mscrobotics2324laptop9/Downloads','deformed1.jpeg'), cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread(os.path.join('/home/mscrobotics2324laptop9/Downloads','deformed1.jpeg'))
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
    epsilon = 0.09 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    #cv2.drawContours(image1, [contour], -1, (0, 0, 255), 2)
    #if cv2.isContourConvex(approx):
     #   cv2.drawContours(image1, [contour], -1, (0, 0, 255), 2)


    if len(approx)==4 and cv2.isContourConvex(approx) and cv2.contourArea(approx)>=60000:
        #cv2.drawContours(image1, [contour], -1, (0, 0, 255), 2)
        print (approx)
        max_cosine = 0
        print (approx)
        ids= find_corner_points_in_contour(approx,contour) 
        print (ids)
        print(find_deformed_sides_idx(contour,ids))
        print(cv2.contourArea(approx))
        for point in approx:
            # Extract the x and y coordinates of the point
            corner = point[0]
            (x,y)=point[0]
            cor= CornerRefinement.refine_corner_harris(image1,corner,10)
            point[0]= cor
            cv2.drawContours(image1, [approx], -1, (0, 0, 255), 2)

            cv2.circle(image1, cor, 1, (0, 255, 0), -1) 
            cv2.putText(image1, f'({x},{y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            #cv2.circle(image1, corner, 1, (0, 0, 255), -1)
        
        success, warped_image = warp_cylinder(image, (49, 49), approx, contour)
        if success:
            cv2.imwrite("Warped_Image.png ", warped_image)

#draw_quads(image1, squares)
markers = []
c1= np.array(contour_try, dtype=np.float32).reshape(-1, 1, 2)
#cv2.drawContours(image1, c1, -1, (0, 0, 255), 2)


cv2.imwrite('squares_output_get_marker.jpg', image1)
end_time_1 = time.time()
elapsed_time_1 = end_time_1 - start_time_1
print(elapsed_time_1)
marker_candidate = MarkerCandidate(contour=[[[258,  23]],

 [[257,  34]],

 [[270,  36]],

 [[271,  25]]], idx=0)

#contour1 = np.array(contour_try, dtype=np.float32).reshape(-1, 1, 2)

#print (contour1)
#success, warped_image = warp_cylinder(image, (300, 300), contour1)

#if success:
    #cv2.imwrite("Warped_Image.png ", warped_image)
    #cv2.waitKey(0)
#else:
    #print("Warping failed.")
