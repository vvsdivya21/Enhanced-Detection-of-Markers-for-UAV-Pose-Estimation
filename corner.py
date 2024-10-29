import cv2
import numpy as np


class CornerRefinement:
    def preprocess_image(image):
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Apply CLAHE for contrast enhancement
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        return equalized
    
    def refine_corner_sobel(gray, corner, box=10):
        gray = preprocess_image(gray)  # Preprocess image
        roi = CornerRefinement.get_roi(gray, corner, box)

        # Apply Sobel operator
        sobel_x = cv2.Sobel(gray[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]], cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]], cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)

        # Find the maximum value in the Sobel image
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sobel)
        x, y = max_loc

        # Optionally display debugging information
        # cv2.rectangle(gray, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
        # cv2.circle(gray, (x + roi[0], y + roi[1]), 3, (0, 255, 0), -1)
        # cv2.circle(gray, (roi[0] + box // 2, roi[1] + box // 2), 3, (0, 0, 255), -1)
        # cv2.imshow('Corner', gray)
        # cv2.waitKey(0)

        return (corner[0] + x - box // 2, corner[1] + y - box // 2)


    def refine_corner_harris(frame, corner, box=10):
        roi = CornerRefinement.get_roi(frame, corner, box)
        area = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]].copy()
        
        gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        #gray = preprocess_image(gray)  # Preprocess image
        dst = cv2.cornerHarris(gray, 2, 3, 0.02)
        dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)

        x, y = corner
        min_distance = box

        for j in range(dst.shape[0]):
            for i in range(dst.shape[1]):
                if dst[j, i] > 150:
                    distance = np.sqrt((j - box // 2) ** 2 + (i - box // 2) ** 2)
                    if distance < min_distance:
                        x, y = i, j
                        min_distance = distance

                    area[j, i] = [0, 255, 0]
                    #cv2.circle(area, (i,j), 2, (0, 255, 0), -1)

        area[box // 2, box // 2]
        #cv2.circle(area, center, 2, (0, 0, 255), -1)
        #cv2.imwrite('fun_ch.png',area)
        # cv2.imshow('Harris', area) 
        # cv2.waitKey(0)

        return (corner[0] + x - box // 2, corner[1] + y - box // 2)

    
    def get_roi(image, center, box):
        x, y = center
        x1 = max(x - box // 2, 0)
        y1 = max(y - box // 2, 0)
        x2 = min(x1 + box, image.shape[1])
        y2 = min(y1 + box, image.shape[0])

        return (x1, y1, x2 - x1, y2 - y1)
    
    def refine_corner_harris_with_adaptive_threshold(frame, corner, box=10):
        gray = preprocess_image(gray)  # Preprocess image
        roi = CornerRefinement.get_roi(frame, corner, box)
        area = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]].copy()
        
        gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, 2, 3, 0.02)
        dst = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
        
        max_dst = np.max(dst)
        thresh = 0.01 * max_dst  # Adaptive thresholding based on max value
        
        x, y = corner
        min_distance = box

        for j in range(dst.shape[0]):
            for i in range(dst.shape[1]):
                if dst[j, i] > thresh:
                    distance = np.sqrt((j - box // 2) ** 2 + (i - box // 2) ** 2)
                    if distance < min_distance:
                        x, y = i, j
                        min_distance = distance

                    area[j, i] = [0, 255, 0]

        area[box // 2, box // 2] = [0, 0, 255]
        # cv2.imshow('Harris', area)
        # cv2.waitKey(0)

        return (corner[0] + x - box // 2, corner[1] + y - box // 2)
    
    def refine_corner_canny(gray, corner, box=10):
        roi = CornerRefinement.get_roi(gray, corner, box)
        area = gray[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]].copy()

        # Apply Canny edge detection
        edges = cv2.Canny(area, 50, 150)
         
        # Find the location of the maximum gradient in the Canny edge map
        y_coords, x_coords = np.nonzero(edges)
        if len(x_coords) == 0:
            return corner  # No edges detected, return original corner

        # Find the point with the maximum gradient
        edge_points = list(zip(x_coords, y_coords))
        distances = [np.sqrt((x - box // 2) ** 2 + (y - box // 2) ** 2) for x, y in edge_points]
        min_index = np.argmin(distances)
        x, y = edge_points[min_index]

        area[y, x] = [0, 255, 0]  # Mark detected edge point
        #cv2.circle(area, center, 2, (0, 255, 0), -1)
        # Display the result (for debugging)
        # cv2.imshow('Canny Edge Detection', area)
        # cv2.waitKey(0)

        return (corner[0] + x - box // 2, corner[1] + y - box // 2)
    
    def refine_corner_combined_harris_sobel(frame, corner, box=10):
        # Use Harris for initial estimation
        initial_corner = CornerRefinement.refine_corner_harris(frame, corner, box)
        # Use Sobel for refinement
        refined_corner = CornerRefinement.refine_corner_sobel(frame, initial_corner, box)
        return refined_corner
    
    def refine_corner_combined_canny_harris(frame, corner, box=10):
        # Use Canny for initial edge detection
        edges = CornerRefinement.refine_corner_canny(frame, corner, box)
        # Use Harris or Sobel for further refinement if needed
        refined_corner = CornerRefinement.refine_corner_harris(frame, edges, box)
        return refined_corner
    
    def detect_good_features(gray, max_corners=100, quality_level=0.01, min_distance=10):
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
        if corners is not None:
            return [tuple(pt[0]) for pt in corners]
        return []



filename = '1023.png'
img = cv2.imread(filename)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cor= CornerRefinement.refine_corner_harris(img,(99,246),10)
cv2.imwrite('fun_ch.png',img)
print (cor)
'''
#find Harris corners
gray = np.float32(gray)

dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)

ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
cv2.imwrite('threshed.png',dst)
dst = np.uint8(dst)

#find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

#define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
#here u can get corners
print (corners)

#Now draw them
res = np.hstack((centroids,corners)) 
res = np.int0(res) 
img[res[:,1],res[:,0]]=[0,0,255] 
#img[res[:,3],res[:,2]] = [0,255,0]
cv2.imwrite('1.png',img)'''



