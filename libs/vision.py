import cv2
import numpy as np
from skimage import feature, measure
from scipy.spatial import distance
import os
from heapq import heappush, heappop


def get_image_from_camera(cam,distorsion=False):
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image")
        return None
    if distorsion:
        return correct_camera_perspective(frame)
    else:
        return frame


def correct_camera_perspective(img):
    try:
        mtx = np.load("grid/camera_matrix.npy")
        dist = np.load("grid/distortion_coefficients.npy")
    except FileNotFoundError:
        print("Calibration files not found.")
        exit()

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image borders
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst

def get_image_from_file(image_path: str)-> np.ndarray:
    """
    Input: filepath of an image
    Output: image as numpy array
    Example: get_image(os.path.join("..", "robot-env", "s1.png"))
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    try:
        img = cv2.imread(image_path) # BGR
        if img is None:
            raise ValueError(f"Unable to read the image from file: {image_path}")
        return img
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


##################################################################################################


def largest_cc(mask: np.ndarray)->np.ndarray:
    """
    Input: an mask
    Output: mask of the largest connected component
    """
    labels = measure.label(mask)
    counts = np.bincount(labels.ravel())
    counts[0] = 0 # disregard background
    largest_label = counts.argmax()
    lcc_mask = labels == largest_label
    return lcc_mask

def find_corners(lcc_mask: np.ndarray, epsilon: float, eps_security=True, verbose=True)->np.ndarray:
    """
    Input: mask of the edges largest connected component
    Output: coordinates of the 4 corners using 
    """ 
    contours, _ = cv2.findContours((lcc_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    approx = cv2.approxPolyDP(largest_contour, epsilon * cv2.arcLength(largest_contour, True), True)
    
    if eps_security:
        while len(approx) != 4:
            epsilon = epsilon + 0.001
            if verbose:
                print(f"Incremented epsilon: {epsilon}")
            approx = cv2.approxPolyDP(largest_contour, epsilon * cv2.arcLength(largest_contour, True), True)
            
    corners = approx.reshape(-1, 2)
    if verbose:
        print(f"Detected {len(corners)} corners:")
        print(corners)

    return corners

def order_points(corners):
    rect = np.zeros((4, 2), dtype="float32")
    
    _sum = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    rect[0] = corners[np.argmin(_sum)]
    rect[2] = corners[np.argmax(_sum)]
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    
    return rect

def compute_destination_size(ordered_corners):
    (top_left, top_right, bottom_right, bottom_left) = ordered_corners
    
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(top_left - bottom_left)
    height_right = np.linalg.norm(top_right - bottom_right)
    max_height = max(int(height_left), int(height_right))

    return max_width, max_height

def correct_perspective(image: np.ndarray, sigma=5, epsilon=0.01, eps_security=True, verbose=False) -> np.ndarray:
    edges = feature.canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sigma=sigma)
    mask = largest_cc(edges)
    corners = find_corners(mask, epsilon=epsilon, eps_security=eps_security, verbose=verbose)
    ordered_corners = order_points(corners)
    max_width, max_height = compute_destination_size(ordered_corners)
    destination_corners = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered_corners, destination_corners)
    corrected_image = cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_LINEAR)
    return corrected_image


##################################################################################################
def full_detection_cnt_centroid(image: np.ndarray, thresh_Thymio, thresh_obstacle, thresh_goal,min_size) -> np.ndarray:
    thresholded_img = np.zeros_like(image)

    ##Find Thymio (start)
    Thymio_mask = cv2.inRange(image, thresh_Thymio[:3], thresh_Thymio[3:6])
    contours, _ = cv2.findContours(Thymio_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]  # biggest contour
    filled_mask = np.zeros_like(Thymio_mask)
    Thymio_mask=cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    thresholded_img[Thymio_mask==255] = [255, 255, 255]

    # Find the minimum enclosing circle
    (_, _), radius = cv2.minEnclosingCircle(cnt)
    radius = int(np.ceil(radius)) # exact radius was too small

    # Find Obstacles
    obstacle_mask=255*np.ones_like(Thymio_mask, dtype=np.uint8)
    for i in range(thresh_obstacle.shape[0]):
        temp_mask = cv2.inRange(image, thresh_obstacle[i,:3], thresh_obstacle[i,3:6])
        obstacle_mask = cv2.bitwise_and(obstacle_mask, temp_mask)
    obstacle_mask = filter_small_blobs(obstacle_mask, min_size=min_size)

    obstacle_mask, obstacle_cnt = fill_holes(obstacle_mask)

    distance = cv2.distanceTransform(~obstacle_mask, cv2.DIST_L2, 5)

    # Threshold to expand the obstacle by a fixed distance (e.g., 10 pixels)
    expanded_obstacle_mask = (distance < radius) * 255

    expanded_obstacle_mask, obstacle_cnt_expnded = fill_holes(expanded_obstacle_mask)
    thresholded_img[expanded_obstacle_mask==255] = [0, 0, 255]


    
    
    # Find Goal
    goal_mask = cv2.inRange(image, thresh_goal[:3], thresh_goal[3:6])

    goal_mask = filter_small_blobs(goal_mask, min_size=min_size)

    goal_mask, goal_cnt = fill_holes(goal_mask)

    thresholded_img[goal_mask==255] = [0, 255, 0]
    #Goal Center:
    M = cv2.moments((goal_mask*255).astype(np.uint8))
    Goal_x = int(M["m10"] / M["m00"])
    Goal_y = int(M["m01"] / M["m00"])
    Goal_center=np.array([Goal_x,Goal_y]).reshape(2,1)

    return thresholded_img,obstacle_cnt, obstacle_cnt_expnded, goal_cnt, Goal_center

def filter_small_blobs(red_mask: np.ndarray, min_size: int) -> np.ndarray:
        if min_size == 1:
            return red_mask
        out_mask = np.zeros_like(red_mask)
        labels = measure.label(red_mask)
        for label in np.unique(labels):
            if label == 0:
                continue
            component = labels == label
            if np.sum(component) >= min_size:
                out_mask[component] = 255
        return out_mask


def filter_big_blobs(red_mask: np.ndarray, max_size: int) -> np.ndarray:
        out_mask = np.zeros_like(red_mask)
        labels = measure.label(red_mask)
        for label in np.unique(labels):
            if label == 0:
                continue
            component = labels == label
            if np.sum(component) <= max_size:
                out_mask[component] = 255
        return out_mask

def fill_holes(bool_mask: np.ndarray)-> np.ndarray:
    mask = (bool_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask,contours

##################################################################################################


def get_dominant_color(block:np.ndarray, verbose: bool)->np.ndarray:
    pixels = block.reshape(-1, 3)

    has_white = np.any(np.all(pixels == [255, 255, 255], axis=1))
    has_green = np.any(np.all(pixels == [0, 255, 0], axis=1))
    has_red = np.any(np.all(pixels == [0, 0, 255], axis=1))

    if has_red:
        return np.array([0, 0, 255], dtype=np.uint8)
    elif has_green:
        return np.array([0, 255, 0], dtype=np.uint8)    
    elif has_white:
        return np.array([255, 255, 255], dtype=np.uint8)
    else:
        return np.array([0, 0, 0], dtype=np.uint8) 


def discretize_image(image:np.ndarray, grid_size: int, verbose: bool, full_output: bool):
    """
    Discretizes an OpenCV image using a grid of grid_size x grid_size cells.
    """
    height, width, _ = image.shape
    cell_height = height // grid_size
    cell_width = width // grid_size

    grid_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    if full_output:
        discretized_image = np.copy(image)

    for i in range(grid_size):
        for j in range(grid_size):
            start_y = i * cell_height
            end_y = (i + 1) * cell_height
            start_x = j * cell_width
            end_x = (j + 1) * cell_width

            block = image[start_y:end_y, start_x:end_x]

            dominant_color = get_dominant_color(block, verbose)

            grid_image[i, j] = dominant_color
            if full_output:
                discretized_image[start_y:end_y, start_x:end_x] = dominant_color

    if full_output:
        return grid_image, discretized_image
    else:
        return grid_image
    
def image_to_grid(grid_image: np.ndarray) -> np.ndarray:
    grid = np.zeros((grid_image.shape[0], grid_image.shape[1]), dtype=np.int8)
    
    background_mask = np.all(grid_image == [0, 0, 0], axis=-1)
    start_mask = np.all(grid_image == [255, 255, 255], axis=-1)
    goal_mask = np.all(grid_image == [0, 255, 0], axis=-1)
    obstacle_mask = np.all(grid_image == [0, 0, 255], axis=-1)
    
    grid[background_mask] = 1
    grid[start_mask] = 1
    grid[goal_mask] = 1
    grid[obstacle_mask] = -1
    
    return grid


def grid_to_image(grid: np.ndarray) -> np.ndarray:
    grid_image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    
    grid_image[grid == 1] = [0, 0, 0] # background/empty
    grid_image[grid == 1] = [255, 255, 255] # start/robot
    grid_image[grid == 1] = [0, 0, 0] # goal
    grid_image[grid == -1] = [0, 0, 255] # obstacle   
    
    return grid_image

def get_grid(image: np.ndarray, grid_size=100, verbose=True, full_output=False)->np.ndarray:
    grid_image = discretize_image(image, grid_size, verbose, full_output)
    grid = image_to_grid(grid_image)
    return grid


##################################################################################################

def get_centroids(grid:np.ndarray, _object):
    
    object_code_map = {
        "obstacle": -1,
        "start": 0,
        "robot": 0,
        "goal": -2,
    }
    
    is_image = len(grid.shape) == 3 and grid.shape[2] == 3

    if isinstance(_object, str):
        if not is_image:
            if _object not in object_code_map:
                raise ValueError(f"Invalid object type string: {_object}. Must be one of {list(object_code_map.keys())}.")
            object_code = object_code_map[_object]
        else:
            raise ValueError("String object types are only supported for grid inputs.")
    elif is_image and isinstance(_object, (list, tuple, np.ndarray)) and len(_object) == 3:
        object_color = np.array(_object)
    else:
        object_code = _object

    if is_image:
        object_mask = np.all(grid == object_color, axis=-1)
    else:
        object_mask = (grid == object_code)
    

    labeled_array = measure.label(object_mask)
    centroids = []
    for region in measure.regionprops(labeled_array):
        y, x = region.centroid
        centroids.append([int(x), int(y)])

    centroids = np.array(centroids)
    if centroids.shape[0] == 1:
        return centroids.reshape(1, 2)
    
    return centroids

def find_nose_corners(image: np.ndarray, sigma, threshold=100, minLineLength=100, maxLineGap=200)->np.ndarray:
    edges = feature.canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sigma=sigma)
    mask = largest_cc(edges).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    lines_mask = np.zeros_like(edges).astype(np.uint8) * 255
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_mask, (x1, y1), (x2, y2), 255, thickness=1)

    curved_border = mask - lines_mask
    curved_border = largest_cc(curved_border).astype(np.uint8) * 255
    contours, _ = cv2.findContours(curved_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    
    contour = max(contours, key=cv2.contourArea)
    
    max_dist = 0
    border_point_1, border_point_2 = None, None
    
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            p1 = contour[i][0]
            p2 = contour[j][0]
            dist = distance.euclidean(p1, p2)
            if dist > max_dist:
                max_dist = dist
                border_point_1, border_point_2 = p1, p2
    
    return curved_border, border_point_1, border_point_2

def get_midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    return np.array([(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2], dtype=int)

def get_slope_intercept(point1, point2):
    if point1[0] == point2[0]: 
        raise ValueError(f"Vertical line detected, cannot compute slope = ({point1}, {point2})")
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    intercept = point1[1] - slope * point1[0]
    return slope, intercept

def get_nose(image:np.array, sigma=5, threshold=25, minLineLength=20, maxLineGap=50):
    curve, bp1, bp2 = find_nose_corners(image, sigma=sigma, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    mid = get_midpoint(bp1, bp2)
    centroid = get_centroids(image, [255,255,255])
    slope, intercept = get_slope_intercept(mid, centroid.flatten())
    if slope is None:
        x = mid[0]
        for y in range(curve.shape[0]):
            if curve[y, x] > 0:
                return np.array([x, y], dtype=int)
    
    contour_points = np.argwhere(curve > 0)
    
    nose = None
    for point in contour_points:
        x, y = point[1], point[0]
        if abs(y - (slope * x + intercept)) < 1:
            nose = np.array([x, y], dtype=int)
    
    return nose

def get_orientation(nose, centroid):
    dx = nose[0] - centroid[0]
    dy = nose[1] - centroid[1]
    
    theta = np.arctan2(dy, dx)
    
    # [0, 2π]
    if theta < 0:
        theta += 2 * np.pi
    
    theta_degrees = np.degrees(theta)
    
    return theta, theta_degrees

##################################################################################################

def heuristic(a, b):
    #Euclidian Distance
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def a_star_search(map_grid, start, goal):

    #Initialize the open set as a priority queue and add the start node
    start=tuple(start.flatten())
    goal=tuple(goal.flatten())

    open_set = []
    heappush(open_set,(0+heuristic(start,goal),0,start))

    came_from = {}
    g_costs = {start: 0}
    explored = set()
    cost_map=-1*np.ones_like(map_grid,dtype=np.float64)

    while open_set:  # While the open set is not empty

        current_f_cost, current_g_cost, current_pos = heappop(open_set)
        # Add the current node to the explored set
        explored.add(current_pos)

        # Check if the goal has been reached
        if current_pos == goal:
            break
        # Get the neighbors of the current node 8 neighbors
        neighbors = [(current_pos[0],current_pos[1]+1),
                     (current_pos[0],current_pos[1]-1),
                     (current_pos[0]-1,current_pos[1]),
                     (current_pos[0]+1,current_pos[1]),
                     (current_pos[0]+1,current_pos[1]+1),
                     (current_pos[0]-1,current_pos[1]-1),
                     (current_pos[0]-1,current_pos[1]+1),
                     (current_pos[0]+1,current_pos[1]-1),

            ]

        for neighbor in neighbors:
            if neighbor in explored:
                continue
            # Check if neighbor is within bounds and not an obstacle
            if (0 <= neighbor[0] < map_grid.shape[0]) and (0 <= neighbor[1] < map_grid.shape[1]) and (map_grid[neighbor]!=-1):
                
                # Calculate tentative_g_cost
                if (neighbor[0]==current_pos[0]) or (neighbor[1]==current_pos[1]): #if goes straight
                    tentative_g_cost = g_costs[current_pos]+(map_grid[neighbor]) #cost is 1 y default on the map_grid
                else:
                    tentative_g_cost = g_costs[current_pos]+(map_grid[neighbor])*np.sqrt(2) #going diagonnaly is going further

                # If this path to neighbor is better than any previous one
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    # Update came_from and g_costs
                    came_from[neighbor] = current_pos
                    g_costs[neighbor] = tentative_g_cost
                    f_cost=tentative_g_cost+heuristic(neighbor,goal)
                    cost_map[neighbor]=f_cost
                    # Add neighbor to open set
                    heappush(open_set, (f_cost,tentative_g_cost,neighbor))
    
    # Reconstruct path
    if current_pos == goal:
        #Reconstruct the path
        path=[goal]
        while path[-1]!=start:
            path.append(came_from[path[-1]])
        return np.array(path[::-1]).T, explored, cost_map  # Return reversed path, explored cells and cost_map for visualization
    else:
        print("Error, no path")
        return None, explored, cost_map
    
##################################################################################################
def grid1_coord2grid2_coord(coord,grid1,grid2):
    coord_grid=np.copy(coord)
    if coord.ndim == 1:
        coord_grid[0]=coord[0]*(grid2.shape[0]-1)/(grid1.shape[0]-1)
        coord_grid[1]=coord[1]*(grid2.shape[1]-1)/(grid1.shape[1]-1)
    else:
        coord_grid[0,:]=coord[0,:]*(grid2.shape[0]-1)/(grid1.shape[0]-1)
        coord_grid[1,:]=coord[1,:]*(grid2.shape[1]-1)/(grid1.shape[1]-1)
    return np.int32(np.rint(coord_grid))

def init(cam, sigma = 5, epsilon = 0.01, thresh_Thymio=np.array([190,190,190,255,255,255]),
        thresh_obstacle=np.array([0,0,120,0,0,140]), thresh_goal=np.array([0,120,0,0,140,0]), min_size=5000, grid_size=200):
    
    #image = get_image_from_camera(cam,False) # camera calibration inside

    #image = correct_perspective(image, sigma=sigma, epsilon=epsilon) TOBEREMOVED
    image=cv2.imread("goodimg.jpg")

    image_colored ,obstacle_cnt, obstacle_cnt_expnded, goal_cnt, Goal_center= full_detection_cnt_centroid(image, thresh_Thymio, thresh_obstacle, thresh_goal, min_size)

    grid = get_grid(image_colored, grid_size, verbose=True, full_output=False)

    grid_image = grid_to_image(grid)

    Thymio_size=-1
    Thymio_xytheta, Thymio_detected, Thymio_size, Thymio_nose, Thymio_cnt = Thymio_position(image, thresh_Thymio, Thymio_size)

    #nose = get_nose(grid_image, sigma=sigma, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)


    #Careful! Image frame's first coord (x) is pointing right but in a matrix the first coordinate (rows) is pointing down so they must be inverted
    path, explored, cost_map = a_star_search(grid, grid1_coord2grid2_coord(np.array([Thymio_xytheta[1],Thymio_xytheta[0]]),image,grid), grid1_coord2grid2_coord(np.array([Goal_center[1],Goal_center[0]]),image,grid))

    return image, grid, Thymio_xytheta, Goal_center, path, Thymio_detected, Thymio_nose, obstacle_cnt, obstacle_cnt_expnded, goal_cnt, Thymio_cnt, Thymio_size, Thymio_nose.reshape(2,1)

def Thymio_position(img, thresh_Thymio, Thymio_size):

    white_mask = cv2.inRange(img, thresh_Thymio[:3], thresh_Thymio[3:6])

    if Thymio_size<0:
        cnt, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #or SIMPLE
        cnt = sorted(cnt, key=cv2.contourArea, reverse=True) #sort by largest cnt, there should be only one with small blob removal but we never know :)
        Thymio_size=cv2.contourArea(cnt[0]) #The biggest is assumed to be the Thymio


    white_mask=filter_big_blobs(white_mask,Thymio_size*1.5)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]  # biggest contour
    Thymio_mask=np.zeros_like(white_mask,dtype=np.uint8)
    cv2.drawContours(Thymio_mask, [cnt], -1, 255, thickness=cv2.FILLED)


    if (cv2.contourArea(cnt)<Thymio_size*0.75):
        Thymio_x=-1000
        Thymio_y=-1000
        Thymio_theta=0
        Thymio_nose=np.array([-1100,-1100])
        Thymio_detected=False
        print("Thymio not detected")
        return Thymio_x, Thymio_y, Thymio_theta, Thymio_detected, Thymio_size, Thymio_nose.reshape(2,1), cnt
    else: Thymio_detected=True
    #Get Centroid
    M = cv2.moments(Thymio_mask)
    Thymio_x = int(M["m10"] / M["m00"])
    Thymio_y = int(M["m01"] / M["m00"])

    #Get orientation 
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect) #get oriented box vertices
    box = np.int32(box) #get integer

    #Get direction
    box_mask=np.zeros_like(Thymio_mask,dtype=np.uint8)
    cv2.drawContours(box_mask, [box], -1, 255, thickness=cv2.FILLED)
    intersection= cv2.bitwise_and(box_mask, ~Thymio_mask) #(box_mask & ~Thymio_mask)*255 manual but less opti
    M = cv2.moments(intersection)
    cX_nose = int(M["m10"] / M["m00"])
    cY_nose = int(M["m01"] / M["m00"])

    Thymio_theta=np.atan2(cY_nose-Thymio_y,cX_nose-Thymio_x)
    Thymio_nose=np.array([cX_nose,cY_nose])

    Thymio_xytheta=np.array([[Thymio_x],[Thymio_y],[Thymio_theta]])
    return Thymio_xytheta, Thymio_detected, Thymio_size, Thymio_nose.reshape(2,1), cnt

def update_vision(cam, sigma = 5, epsilon = 0.01, T_WL=190, Thymio_size=-1):
    
    image = get_image_from_camera(cam)
    
    image = correct_perspective(image, sigma=sigma, epsilon=epsilon)

    # detection thymio
    Thymio_xytheta, Thymio_detected,Thymio_size, Thymio_nose, Thymio_cnt = Thymio_position(image, T_WL, Thymio_size)

    return  Thymio_xytheta, Thymio_detected 