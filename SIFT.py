import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
'''    {
        'id': 2,
        'image': cv2.imread('marker_0004.png'),  # Replace with your marker image path
        'keypoints': None,
        'descriptors': None
    },
    {
        'id': 3,
        'image': cv2.imread('marker_0010.png'),  # Replace with your marker image path
        'keypoints': None,
        'descriptors': None
    }'''
predefined_markers = [
    {
        'id': 1,
        'imagep': cv2.imread('marker_aruco_original0100.png'),  # Replace with your marker image path
        'keypoints': None,
        'descriptors': None
    },
    {
        'id': 2,
        'imagep': cv2.imread('marker_aruco_original0123.png'),  # Replace with your marker image path
        'keypoints': None,
        'descriptors': None
    },
    {
        'id': 3,
        'imagep': cv2.imread('marker_aruco_original0125.png'),  # Replace with your marker image path
        'keypoints': None,
        'descriptors': None
    }

]
def detect_features(image):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    if img1.dtype != 'uint8':
        img1 = cv2.convertScaleAbs(img1)
    if img2.dtype != 'uint8':
        img2 = cv2.convertScaleAbs(img2)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches
    #cv2.imshow('Matches', img_matches)
def draw_marker_detection(base_img, marker_img, H, marker_kp, matches):
    # Project marker corners onto the base image
    h, w = marker_img.shape[:2]
    marker_corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(marker_corners, H)
    
    # Draw the detected marker corners
    base_img_with_corners = cv2.polylines(base_img, [np.int32(projected_corners)], True, (0, 255, 0), 2, cv2.LINE_AA)
    return base_img_with_corners, projected_corners

gray= cv2.imread('marker_aruco_original0100.png', cv2.IMREAD_GRAYSCALE)
image= cv2.imread('marker_aruco_original0100.png')
im2=cv2.imread(os.path.join('/home/mscrobotics2324laptop9/Downloads','4.jpeg'), cv2.IMREAD_GRAYSCALE)
image2=cv2.imread(os.path.join('/home/mscrobotics2324laptop9/Downloads','4.jpeg'))

# Initialize the SIFT detector
sift = cv2.SIFT_create(100000)

# Detect keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)
kp2,dsc2=sift.detectAndCompute(im2,None)    
matches = match_features(descriptors, dsc2)
img_matches = draw_matches(image, keypoints, image2, kp2, matches)
cv2.imwrite(os.path.join('/home/mscrobotics2324laptop9/images_dissertation','sift4.jpg'), img_matches)

src_pts = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimate homography
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
a, detected_corners=draw_marker_detection(image2, image, H, kp2, matches)
cv2.imwrite(os.path.join('/home/mscrobotics2324laptop9/images_dissertation','sift_marker4.jpg'),a)

# Draw keypoints on the image
#scene_image_with_corners, detected_corners = cv2.drawKeypoints(im2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

for corner in detected_corners:
    print(corner[0])  # Each corner is in a nested array


#cv2.imwrite(os.path.join('/home/mscrobotics2324laptop9/images_dissertation','SIFT_corners1.jpg'), scene_image_with_corners)





'''for marker in predefined_markers:
        marker['keypoints'], marker['descriptors'] = detect_features(marker['imagep'])
        matches = match_features(descriptors, marker['descriptors'])
        img_matches = draw_matches(marker['imagep'], marker['keypoints'], image2, kp2, matches)
        cv2.imwrite('SIFT_Keypoints3.jpg', img_matches)'''



# Display the image with keypoints using OpenCV
#cv2.imwrite('SIFT_Keypoints.jpg', img_matches)
'''cv2.waitKey(0)
cv2.destroyAllWindows()

# If you prefer to use matplotlib for display (especially useful in notebooks)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.axis('off')
plt.show()


'''
'''
# Parameters
limit_cosine = 0.6
min_area = 0
max_error = 0.02

# Function to draw quadrilaterals
def draw_quads(mat, quads):
    for quad in quads:
        for j in range(4):
            cv2.line(mat, tuple(quad.points[j]), tuple(quad.points[(j + 1) % 4]), (255, 0, 255), 2)

# Function to calculate the cosine of the angle between corner points
def angle_corner_points_cos(b, c, a):
    dx1 = b[0] - a[0]
    dy1 = b[1] - a[1]
    dx2 = c[0] - a[0]
    dy2 = c[1] - a[1]
    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude1 = dx1 * dx1 + dy1 * dy1
    magnitude2 = dx2 * dx2 + dy2 * dy2
    epsilon = 1e-12  # To avoid division by zero
    denom = np.sqrt(max(magnitude1 * magnitude2, epsilon))
    
    if denom == 0:
        print(f"Zero denominator encountered with points a={a}, b={b}, c={c}")
        return 0
    
    return dot_product / denom

# Function to detect features using SIFT
def detect_features(image):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Function to match features using BFMatcher
def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Function to estimate the pose using homography
def estimate_pose(matches, keypoints1, keypoints2):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M, mask

# Function to draw matches between the current frame and the marker
def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)

# Load predefined markers (example using one marker for simplicity)
predefined_marker = {
    'id': 1,
    'image': cv2.imread('marker_003.png', cv2.IMREAD_GRAYSCALE),
    'keypoints': None,
    'descriptors': None
}
predefined_marker['keypoints'], predefined_marker['descriptors'] = detect_features(predefined_marker['image'])

# Start video capture

# Convert the frame to grayscale
frame=cv2.imread('')
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect features in the current frame
keypoints_frame, descriptors_frame = detect_features(gray_frame)

# Match features with predefined marker
matches = match_features(descriptors_frame, predefined_marker['descriptors'])

# Check if enough matches are found
if len(matches) > 10:
    # Estimate pose
    M, mask = estimate_pose(matches, keypoints_frame, predefined_marker['keypoints'])
    
    if M is not None:
        # Draw matches and detected marker on the frame
        draw_matches(gray_frame, keypoints_frame, predefined_marker['image'], 
                    predefined_marker['keypoints'], matches)
        
        # Further processing for detected marker
        # ...

# Display the frame
cv2.imshow('Frame', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()'''
