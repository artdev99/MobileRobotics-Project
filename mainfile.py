import cv2
import numpy as np
from libs.vision import *
from skimage import feature, measure
from scipy.spatial import distance
import os
from heapq import heappush, heappop
import time


start_time = time.time()
cv2.destroyAllWindows()

# Variables initialization
Thymio_xytheta_hist = np.empty((0, 3))
sigma = 5
epsilon = 0.01
thresh_obstacle = np.array([[4, 0, 90, 90, 90, 150]])  # Adjust thresholds if needed
thresh_goal = np.array([30, 30, 20, 90, 120, 80])
min_size = 500
grid_size = 300
Thymio_detected = False

# Uncomment and adjust if using an IP camera
"""
print("Try to open camera")

login = "thymio"
password = "qwertz"
url = f"https://{login}:{password}@192.168.21.126:8080/video"  # Check the URL if needed
cam = cv2.VideoCapture(url)
"""

print("Try to open camera")

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Camera could not be opened")
    cam.release()
    exit()

print("Warming up the camera...")
for _ in range(50):  # Number of frames to discard (adjust if needed)
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture frame")
        break

# Initialization function
image, grid, Thymio_xytheta, c_goal, path, obstacle_cnt, obstacle_cnt_expnded, goal_cnt, \
mat_persp, max_width_persp, max_height_persp, aruco_size = init(
    cam, sigma, epsilon, thresh_obstacle, thresh_goal, min_size, grid_size, aruco=True)

# Convert path coordinates for plotting
path_img = grid1_coord2grid2_coord(path, grid, image)
path_img = path_img[::-1]

image_cnt = draw_cnt_image(
    image, goal_cnt, obstacle_cnt, obstacle_cnt_expnded, path_img, Thymio_xytheta, c_goal, aruco_size)

# Update history
Thymio_xytheta_hist = np.vstack((Thymio_xytheta_hist, Thymio_xytheta))

print(f"Initialization time: {time.time() - start_time:.2f} seconds")

######################################################
# Real-time plot images initialization
######################################################
plot_height, plot_width = 400, 600  # Size of the plot images
max_steps = 50  # Maximum number of steps to display
x_plot_img = np.zeros((plot_height, plot_width, 3), dtype=np.uint8)
y_plot_img = np.zeros((plot_height, plot_width, 3), dtype=np.uint8)
theta_plot_img = np.zeros((plot_height, plot_width, 3), dtype=np.uint8)

######################################################
# UPDATE
######################################################
for steps in range(50):
    start_time = time.time()
    image, Thymio_xytheta, Thymio_detected = update_vision(
        cam, sigma, epsilon, mat_persp, max_width_persp, max_height_persp)
    if not Thymio_detected:
        Thymio_xytheta = Thymio_xytheta_hist[-1, :]

    image_cnt = draw_cnt_image(
        image, goal_cnt, obstacle_cnt, obstacle_cnt_expnded, path_img, Thymio_xytheta, c_goal, aruco_size)

    # Overlay x, y, theta onto the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"x: {Thymio_xytheta[0]:.2f}, y: {Thymio_xytheta[1]:.2f}, theta: {Thymio_xytheta[2]:.2f}"
    cv2.putText(image_cnt, text, (10, 30), font, 0.8, (255, 255, 255), 2)

    # Update history
    Thymio_xytheta_hist = np.vstack((Thymio_xytheta_hist, Thymio_xytheta))
    if Thymio_xytheta_hist.shape[0] > max_steps:
        Thymio_xytheta_hist = Thymio_xytheta_hist[-max_steps:]  # Keep only recent data

    # Update plots
    x_data = Thymio_xytheta_hist[:, 0]
    y_data = Thymio_xytheta_hist[:, 1]
    theta_data = Thymio_xytheta_hist[:, 2]

    x_plot_img = update_plot(x_plot_img, x_data, color=(255, 0, 0), label='x (pixels)')
    y_plot_img = update_plot(y_plot_img, y_data, color=(0, 255, 0), label='y (pixels)')
    theta_plot_img = update_plot(theta_plot_img, theta_data, color=(0, 0, 255), label='theta (rad)')

    # Display the image and plots using cv2.imshow
    cv2.imshow('Camera View', image_cnt)
    cv2.imshow('X Position History', x_plot_img)
    cv2.imshow('Y Position History', y_plot_img)
    cv2.imshow('Theta History', theta_plot_img)

    # Wait for 1 ms; necessary for cv2.imshow to work properly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(f"Time for this step: {time.time() - start_time:.2f} seconds")

# Clean up
cam.release()
cv2.destroyAllWindows()



# Function to update plot images
def update_plot(plot_img, data_hist, color=(0, 255, 0), label=''):
    plot_img[:] = 0  # Clear the image
    num_points = len(data_hist)
    if num_points < 2:
        return plot_img
    # Normalize data to fit the plot
    data = np.array(data_hist)
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min if data_max - data_min != 0 else 1
    normalized_data = (data - data_min) / data_range
    # Map data to image coordinates
    x_coords = np.linspace(0, plot_width, num_points).astype(np.int32)
    y_coords = (plot_height - normalized_data * plot_height).astype(np.int32)
    # Draw the plot
    points = np.vstack((x_coords, y_coords)).astype(np.int32).T
    for i in range(1, len(points)):
        cv2.line(plot_img, tuple(points[i - 1]), tuple(points[i]), color, 2)
    # Add labels
    cv2.putText(plot_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return plot_img
