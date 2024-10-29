import cv2
import numpy as np

# Load the image
image_path = 'marker_aruco_original0100.png'  # Use the correct path to your image
image = cv2.imread(image_path)

# Define the grid size
rows, cols = 7, 7
height, width, _ = image.shape

# Calculate the spacing between grid lines
row_step = height // rows
col_step = width // cols

# Draw the horizontal grid lines
for i in range(1, rows):
    start_point = (0, i * row_step)
    end_point = (width, i * row_step)
    image = cv2.line(image, start_point, end_point, (0, 0, 255), 3)  # Red color line

# Draw the vertical grid lines
for j in range(1, cols):
    start_point = (j * col_step, 0)
    end_point = (j * col_step, height)
    image = cv2.line(image, start_point, end_point, (0, 0, 255), 3)  # Red color line

# Save the result
output_path = '7x7_grid_image.png'
cv2.imwrite(output_path, image)


