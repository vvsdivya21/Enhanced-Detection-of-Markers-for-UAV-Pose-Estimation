import cv2 
import numpy as np
from ArucoMarker import ArucoMarker
# Load the predefined dictionary
#img= cv2.imread('images.jpeg')
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
'''aruco_parameters = cv2.aruco.DetectorParameters()
cor, marker_ids, rejected = cv2.aruco.detectMarkers(
            img, aruco_dict, parameters= aruco_parameters
        )
print (marker_ids)
a=img.copy()
img_with_markers = cv2.aruco.drawDetectedMarkers(a, cor, marker_ids)
cv2.imwrite('detectaruco.png', a)'''
# Function to convert the byte pattern to a binary matrix
# Function to convert byte pattern to a 6x6 bit matrix
def bytes_to_6x6_matrix(byte_pattern):
    # Convert bytes to binary representation (8 bits per byte)
    bits = np.unpackbits(byte_pattern)
    
    # Flatten the bits array
    flat_bits = bits.flatten()
    
    # We need exactly 36 bits for a 6x6 matrix
    if len(flat_bits) < 25:
        raise ValueError("Not enough bits to form a 6x6 matrix.")
    
    # Extract the first 36 bits
    six_by_six_bits = flat_bits[:25]
    
    # Reshape into 6x6 matrix
    bit_pattern_6x6 = six_by_six_bits.reshape(5, 5)
    
    return bit_pattern_6x6
# Provided byte pattern
bit_pattern_6x6=[]
#for marker_id in range (13):
byte_pattern = aruco_dict.bytesList[123]
print (byte_pattern)
bit_pattern_6x6.append(bytes_to_6x6_matrix(byte_pattern))
    #Print the 6x6 bit matrix
print("6x6 Bit Pattern:")
print(bit_pattern_6x6)
