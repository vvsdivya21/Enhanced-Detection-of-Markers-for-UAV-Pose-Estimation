import numpy as np
import matplotlib.pyplot as plt
'''
#Occlusion Percentage vs Error

# Define the baseline and other sets of corner points
baseline_points = np.array([
    [406.651, 942.3293],
    [682.14294, 960.0027],
    [683.91046, 1231.9955],
    [396.6481, 1235.9047]
])

corner_points_0 = np.array([
    [405.651, 940.3293],
    [682.14294, 960.0027],
    [683.91046, 1231.9955],
    [396.6481, 1235.9047]
])

corner_points_1 = np.array([
    [631.0177, 1051.272],
    [704.4228, 885.188],
    [642.86383, 1007.41986],
    [423.5285, 1326.1249]
])

corner_points_2 = np.array([
    [597.1696, 783.2828],
    [593.0863, 788.87396],
    [596.9155, 787.8704],
    [590.2941, 784.6422]
])

corner_points_3 = np.array([
    [555.5309, 1215.7427],
    [528.23584, 1300.0496],
    [553.87256, 1242.8832],
    [492.20032, 1297.0238]
])

# Function to calculate Euclidean distance between two sets of points
def calculate_error_percentage(points, baseline):
    errors = np.sqrt(np.sum((points - baseline) ** 2, axis=1))
    baseline_distances = np.sqrt(np.sum(baseline ** 2, axis=1))
    error_percentages = (errors / baseline_distances) * 100
    return error_percentages

# Calculate error percentages for each set relative to the baseline
error_percentages_0 = calculate_error_percentage(corner_points_0, baseline_points)
error_percentages_1 = calculate_error_percentage(corner_points_1, baseline_points)
error_percentages_2 = calculate_error_percentage(corner_points_2, baseline_points)
error_percentages_3 = calculate_error_percentage(corner_points_3, baseline_points)

# Plotting the error percentages
corner_labels = ['Corner 1', 'Corner 2', 'Corner 3', 'Corner 4']

plt.figure(figsize=(10, 6))

plt.plot(corner_labels, error_percentages_0, marker='o', label='Level 1: 14% occlusion', linestyle=':', color='orange')
plt.plot(corner_labels, error_percentages_1, marker='o', label='Level 2: 28% occlusion', linestyle='-', color='blue')
plt.plot(corner_labels, error_percentages_2, marker='s', label='Level 3: 56% occlusion', linestyle='--', color='red')
plt.plot(corner_labels, error_percentages_3, marker='^', label='Level 4: 77% occlusion', linestyle='-.', color='green')

plt.xlabel('Corner Points')
plt.ylabel('Error Percentage (%)')
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
plt.show()'''

'''
#---------------------------------Fold vs Pose

import matplotlib.pyplot as plt
import numpy as np

# Define the centroids
centroids = {
    '10°': (0.32675, 0.20695),
    '20°': (0.405, 0.292),
    '50°': (0.4285, 0.23145),
    '80°': (0.4145, 0.35295)
}

# Reference point (can be the origin or the first fold level)
#reference_point = centroids['Fold level 1']

# Calculate the Euclidean distances from the reference point
distances = {}
for fold, coord in centroids.items():
    distance = np.sqrt((coord[0])**2 + (coord[1])**2)
    distances[fold] = distance

# Prepare data for plotting
fold_levels = list(distances.keys())
distance_values = list(distances.values())

# Create the plot
fig, ax = plt.subplots()
ax.plot(fold_levels, distance_values, marker='o', linestyle='-', color='blue')

# Annotate each point with the distance value
#for i, distance in enumerate(distance_values):
#    ax.text(i, distance, f'{distance:.4f}', color='black', ha='center', va='bottom')

for i, (fold, distance) in enumerate(distances.items()):
    ax.text(i, distance, f'{distance:.4f}', color='black', ha='center', va='bottom')

# Set labels and title
ax.set_xlabel('Fold Levels')
ax.set_ylabel('Translational Distance to the Marker (m)')


# Show the plot
plt.grid(True)
plt.show()
'''


#-------------------------------------------distortion vs rmse
import numpy as np
import matplotlib.pyplot as plt

# Define the original points
original_points = np.array([
    [594, 792],
    [825, 809],
    [821, 1076],
    [591, 1082]
])

# Define the true marker points
true_points = np.array([
    [616.32806, 671.2239],
    [828.8063, 761.9424],
    [816.28546, 953.66644],
    [615.44556, 973.49426]
])

# Define additional point sets
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
    rmse = np.sqrt(np.mean(euclidean_distances**2))
    return rmse

# Initialize specific RMSE values
error0= calculate_rmse(set2,set2) #16
error1= calculate_rmse(true_points,original_points) #12
error2= calculate_rmse(set2,set1)#8import matplotlib.pyplot as plt

# Data points
distortion_levels = [5, 7.5, 10, 12.5, 15, 17.5, 20]
rmse_errors = [120, 80, 50, 35, 12, 4, 0]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(distortion_levels, rmse_errors, marker='o', linestyle='-', color='b', label='RMSE Error vs. Distortion Level')

# Set the axis labels and title
plt.xlabel('Distortion Diameter in cm')
plt.ylabel('RMSE Error in pixels')
plt.xlim(1, 20)  # Set x-axis range from 1 to 20
plt.ylim(0, 130)  # Set y-axis range to comfortably fit the data points
plt.grid(True)
plt.legend()

# Show the plot
plt.show()



'''
#--------------------TIME
import matplotlib.pyplot as plt

# Data points
new_times = [0.08509278297424316, 0.0159914493560791]
old_times = [0.1680436134338379, 0.25745759010314941]

# Combine data for box plot
data = [new_times, old_times]

# Create the box plot
plt.figure(figsize=(5, 5))
plt.boxplot(data, labels=['New Method', 'Old Method'], patch_artist=True, medianprops=dict(color="black"))

# Set the labels and title
plt.xlabel('Method')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Processing Times')

# Show the plot
plt.show()

'''
'''
import matplotlib.pyplot as plt

# Data after swapping Case 1 and Case 2
cases = ['Case 0', 'Case 2', 'Case 1', 'Case 3']
canny_averages = [2.51, 2.51, 5.49, 5.64]
adaptive_averages = [6.96, 7.28, 7.80, 9.09]

# Create a bar chart
fig, ax = plt.subplots()
bar_width = 0.35
index = range(len(cases))

# Plot bars
bars1 = ax.bar(index, canny_averages, bar_width, label='Proposed Method', color='blue')
bars2 = ax.bar([i + bar_width for i in index], adaptive_averages, bar_width, label='ArUco Library', color='red')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Illumination Level')
ax.set_ylabel('Average Error Correction Values')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(cases)
ax.legend()

# Add labels above the bars
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom')

plt.show()
'''