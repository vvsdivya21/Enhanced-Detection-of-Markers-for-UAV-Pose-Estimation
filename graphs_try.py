import numpy as np
import matplotlib.pyplot as plt

# Define the points
original_points = np.array([
    [594, 792],
    [825, 809],
    [821, 1076],
    [591, 1082]
])

true_points = np.array([
    [616.32806, 671.2239],
    [828.8063, 761.9424],
    [816.28546, 953.66644],
    [615.44556, 973.49426]
])

set1 = np.array([
    [394.31128, 1195.0494],
    [646.69684, 1219.0497],
    [667.1942, 1472.9996],
    [373.15762, 1505.2292]
])

set2 = np.array([
    [401, 1178],
    [678, 1179],
    [676, 1460],
    [400, 1457]
])

# Function to calculate RMSE
def calculate_rmse(points1, points2):
    differences = points1 - points2
    euclidean_distances = np.sqrt(np.sum(differences**2, axis=1))
    return np.sqrt(np.mean(euclidean_distances**2))

# Calculate RMSE values
error0 = calculate_rmse(set2, set2)  # Reference RMSE
error1 = calculate_rmse(true_points, original_points)
error2 = calculate_rmse(set2, set1)

# Data points
distortion_levels = [5, 7.5, 10, 12.5, 15, 17.5, 20]
rmse_errors = [120, 80, 50, 35, 12, 4, 0]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(distortion_levels, rmse_errors, marker='o', linestyle='-', color='r')

# Set the axis labels and title
plt.xlabel('Distortion Diameter in cm')
plt.ylabel('RMSE Error in pixels')
plt.xlim(1, 20)
plt.ylim(0, 130)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
