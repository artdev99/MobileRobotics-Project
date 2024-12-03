import cv2
import numpy as np
from skimage import measure

# BGR colors
WHITE = [255,255,255]
RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]
BLACK = [0,0,0]

# Detection
THYMIO_RADIUS = 70 # mm
RADIUS_FACTOR = 0.38
MIN_BLOB_SIZE = 5000
THRESHOLDS_OBSTACLE = np.array([0,0,120,0,0,140])
THRESHOLDS_GOAL = np.array([0,120,0,0,140,0])

class ImageProcessor:
    def __init__(self, thymio_radius = THYMIO_RADIUS, radius_factor = RADIUS_FACTOR, min_blob_size = MIN_BLOB_SIZE, thresholds_obstacle = THRESHOLDS_OBSTACLE, 
                 thresholds_goal = THRESHOLDS_GOAL):
        self.thymio_radius = thymio_radius
        self.radius_factor = radius_factor
        self.min_blob_size = min_blob_size
        self.thresholds_obstacle = thresholds_obstacle
        self.thresholds_goal = thresholds_goal

        self.thresholded_image = None
        self.obstacle_contour = None 
        self.obstacle_contour_expnded = None
        self.goal_contour = None
        self.goal_center = None

    def fill_holes(self, bool_mask):
        mask = (bool_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
        return filled_mask, contours
    
    def filter_small_blobs(self, red_mask):
        if self.min_blob_size == 1:
            return red_mask
        out_mask = np.zeros_like(red_mask)
        labels = measure.label(red_mask)
        for label in np.unique(labels):
            if label == 0:
                continue
            component = labels == label
            if np.sum(component) >= self.min_blob_size:
                out_mask[component] = 255
        return out_mask 

    def detection(self, image, pixbymm):
        # Find Obstacles
        obstacle_mask=255*np.ones(image.shape[:2], dtype=np.uint8)
        for i in range(self.thresholds_obstacle.shape[0]):
            temp_mask = cv2.inRange(image, self.thresholds_obstacle[i,:3], self.thresholds_obstacle[i,3:6]) # to update
            obstacle_mask = cv2.bitwise_and(obstacle_mask, temp_mask)
        obstacle_mask = self.filter_small_blobs(obstacle_mask)
        obstacle_mask, obstacle_contours = self.fill_holes(obstacle_mask)
        distance = cv2.distanceTransform(~obstacle_mask, cv2.DIST_L2, 5)
       
        # Expand the obstacle by Thymio's radius
        thresholded_img = np.zeros_like(image)
        radius= self.radius_factor * self.thymio_radius * pixbymm
        expanded_obstacle_mask = (distance < radius) * 255

        expanded_obstacle_mask, obstacle_contours_expnded = self.fill_holes(expanded_obstacle_mask)
        thresholded_img[expanded_obstacle_mask==255] = RED

        # Find Goal
        goal_mask = cv2.inRange(image, self.thresholds_goal[:3], self.thresholds_goal[3:6]) # to update
        goal_mask = self.filter_small_blobs(goal_mask)
        goal_mask, goal_contour = self.fill_holes(goal_mask)
        thresholded_img[goal_mask==255] = GREEN
        
        M = cv2.moments((goal_mask*255).astype(np.uint8))
        if M["m00"] != 0:
            goal_x = int(M["m10"] / M["m00"])
            goal_y = int(M["m01"] / M["m00"])
            goal_center = np.array([goal_x, goal_y]).reshape(2,1)
        else:
            goal_center = None
            print("Goal not detected")

        return thresholded_img, obstacle_contours, obstacle_contours_expnded, goal_contour, goal_center


