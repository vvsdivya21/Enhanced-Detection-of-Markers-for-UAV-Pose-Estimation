import cv2
import numpy as np
from math import fabs, sqrt
from Quadrilaterals import Quadrilateral
import os

class SquareFinder:
    @staticmethod
    def aspect_ratio(self):
        # Calculate side lengths
        side_lengths = [
            np.linalg.norm(self.points[0] - self.points[1]),
            np.linalg.norm(self.points[1] - self.points[2]),
            np.linalg.norm(self.points[2] - self.points[3]),
            np.linalg.norm(self.points[3] - self.points[0])
        ]
        # Calculate ratios of adjacent sides
        ratio1 = side_lengths[0] / side_lengths[2]
        ratio2 = side_lengths[1] / side_lengths[3]
        return abs(ratio1 - ratio2)
    
    def find_squares(img, limit_cosine=0.6, min_area=0, max_error=0.09):
        squares = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        color = (0, 255, 0)  # Green color for contours
        #cv2.drawContours(color_image, contours, -1, (0, 255, 0), 2)
        
        
        

        contour_count=0
        for contour in contours:
            
            approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * max_error, True)
            
            
            if len(approx) == 4 and fabs(cv2.contourArea(approx)) > 7000 and cv2.isContourConvex(approx):
                max_cosine = 0
                for point in approx:
                    corner=point[0]
                    refined_corner= SquareFinder.refine_corner_harris(img,corner,10)
                    point[0]= refined_corner
                    print (approx)
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

                    #cv2.circle(binary_image,(point[0]),2,(0,0,255),-1)
            
                for j in range(2, 5):
                    cosine = fabs(SquareFinder.angle_corner_points_cos(approx[j % 4][0], approx[j - 2][0], approx[j - 1][0]))
                    max_cosine = max(max_cosine, cosine)

                if max_cosine < limit_cosine:
                    quad = Quadrilateral(
                        approx[0][0],
                        approx[1][0],
                        approx[2][0],
                        approx[3][0]
                    )
                    # Additional geometric validation checks
                    #if quad.aspect_ratio() < 0.2 :
                    squares.append(quad)
        #cv2.imwrite('contoured.jpg',binary_image)
        return squares

    @staticmethod
    def draw_quads(mat, quads):
        for quad in quads:
            for j in range(4):
                cv2.line(mat, tuple(quad.points[j]), tuple(quad.points[(j + 1) % 4]), (255, 0, 255), 2)

    @staticmethod
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
    

    @staticmethod
    def refine_corner_harris(frame, corner, box=10):
        roi = SquareFinder.get_roi(frame, corner, box)
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
        return (corner[0] + x - box // 2, corner[1] + y - box // 2)


    
    def get_roi(image, center, box):
        x, y = center
        x1 = max(x - box // 2, 0)
        y1 = max(y - box // 2, 0)
        x2 = min(x1 + box, image.shape[1])
        y2 = min(y1 + box, image.shape[0])

        return (x1, y1, x2 - x1, y2 - y1)

if __name__ == "__main__":
    # Load an example image
    image = cv2.imread(os.path.join('/home/mscrobotics2324laptop9/Downloads','WhatsApp Image 2024-08-28 at 19.45.07.jpeg'))

    # Find squares in the image
    #contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
 
    #print (contours)
    squares = SquareFinder.find_squares(image)
   
    print(squares)

    # Draw the found squares on the original image
    #color_image1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #SquareFinder.draw_quads(image, squares)

    # Show the image with the drawn squares
    cv2.imwrite(os.path.join('/home/mscrobotics2324laptop9/images_dissertation','fold_out_out_c.jpg'), image)   
