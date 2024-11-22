import cv2
import numpy as np
from skimage import feature, measure
from scipy.spatial import distance
import os
from heapq import heappush, heappop

WHITE = [255,255,255]
RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]
BLACK = [0,0,0]

START = 1
ROBOT = 1
GOAL = -2
OBSTACLE = -1
BACKGROUND = 0

##################################################################################################

def get_object(code):
    if isinstance(code, list):
        if code == WHITE:
            return ROBOT
        if code == GREEN:
            return GOAL
        if code == RED:
            return OBSTACLE
        if code == BLACK:
            return BACKGROUND
    elif isinstance(code, str):
        if code == "robot" or code == "start":
            return ROBOT
        if code == "goal":
            return GOAL
        if code == "obstacle":
            return OBSTACLE
        if code == "background":
            return BACKGROUND
    else:
        ValueError("Color not supported")

def get_color(code):
    if isinstance(code, str):
        if code == "start" or code == "robot":
            return np.array(WHITE, dtype=np.uint8)
        if code == "obstacle":
            return np.array(RED, dtype=np.uint8)
        if code == "goal":
            return np.array(GREEN, dtype=np.uint8)
        if code == "background":
            return np.array(BLACK, dtype=np.uint8)
    elif isinstance(code, int):
        if code == ROBOT:
            return np.array(WHITE, dtype=np.uint8)
        if code == OBSTACLE:
            return np.array(RED, dtype=np.uint8)
        if code == GOAL:
            return np.array(GREEN, dtype=np.uint8)
        if code == BACKGROUND:
            return np.array(BLACK, dtype=np.uint8)
    else:
        raise ValueError("Code not supported")

##################################################################################################

def setup_camera(camera, resolution=None, fps=None):
    """
    1920, 1080 (16:9)
    1280, 720  (16:9)
    640, 380   (16:9)
    320, 240   (4:3)
    """
    if resolution is not None:
        width, height = resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        camera.set(cv2.CAP_PROP_FPS, fps)

    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = int(camera.get(cv2.CAP_PROP_FPS))

    print(f"Resolution is: {w}x{h}")
    print(f"FPS is: {f}") 

def get_image_from_camera(cam, distortion=False, alpha=1, var_treshold=2000):
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image")
        return None
    if camera_obstructed(frame, var_treshold):
        return None
    if distortion:
        frame = correct_camera_distortion(frame, alpha)
    return frame

def correct_camera_distortion(img, alpha):
    try:
        mtx = np.load("libs/camera_matrix_1080.npy")
        dist = np.load("libs/distortion_coefficients_1080.npy")
    except FileNotFoundError:
        print("Calibration files not found")

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))
    
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi # crop the image borders
    dst = dst[y:y+h, x:x+w]

    return dst

def get_image_live(cam, sigma=5, t1=50, t2=150, epsilon=0.1, circle_size=10, border_size=3, resize_factor=0.5):
    if not cam.isOpened():
        print("Error: Camera could not be opened.")
        cam.release()
        exit()

    image = None 
    print("Press 's' to save an image or 'q' to quit.")

    while True:
        ret, frame = cam.read()  
        if not ret:
            print("Failed to grab frame. Exiting.")
            break
        temp = frame.copy()
        temp = cv2.resize(temp, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

        #edges = feature.canny(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY), sigma=sigma)
        blurred_image = cv2.GaussianBlur(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY), (sigma, sigma), 1.4)
        edges = cv2.Canny(blurred_image, t1, t2)
        mask = largest_cc(edges)
        corners = find_corners(mask, epsilon=epsilon, eps_security=False, verbose=False)

        cv2.drawContours(temp, [corners.reshape(-1, 2)], -1, (255, 255, 0), border_size)
        for corner in corners:
            cv2.circle(temp, tuple(corner), circle_size, (255, 0, 255), -1)
        cv2.imshow("Live Feed", temp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  
            image = frame.copy()  
            print("Image saved to variable 'image'.")
        elif key == ord('q'):  
            print("Exiting.")
            break

    #cam.release()
    cv2.destroyWindow("Live Feed")  

    if image is not None:
        print(f"Shape: {image.shape}, pixels {image.shape[0]*image.shape[1]}")

    return image


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


def camera_obstructed(frame, var_treshold=2000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var = np.var(gray)
    return var < var_treshold


##################################################################################################


def largest_cc_old(mask: np.ndarray)->np.ndarray:
    """
    Input: an mask
    Output: boolean mask of the largest connected component
    """
    labels = measure.label(mask)
    counts = np.bincount(labels.ravel())
    counts[0] = 0 # disregard background
    largest_label = counts.argmax()
    lcc_mask = labels == largest_label
    return lcc_mask

def largest_cc(mask: np.ndarray) -> np.ndarray:
    """
    Input: a binary mask
    Output: boolean mask of the largest connected component
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    _, labels = cv2.connectedComponents(mask)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # disregard background
    largest_label = counts.argmax()
    lcc_mask = (labels == largest_label)
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

def correct_perspective(image: np.ndarray, sigma=5, t1=50, t2=150, epsilon=0.01, eps_security=True, verbose=False) -> np.ndarray:
    #edges = feature.canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sigma=sigma)
    blurred_image = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (sigma, sigma), 1.4)
    edges = cv2.Canny(blurred_image, t1, t2)
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


def threshold_color(image, t_image, t_lows, t_highs, color):
    mask = cv2.inRange(image, t_lows, t_highs)
    t_image[mask > 0] = color
    return t_image

def threshold_colors(image: np.ndarray, T_WL: int, T_RH: int, T_RL: int, T_GH: int, T_GL: int) -> np.ndarray:
    thresholded_img = np.zeros_like(image)
    
    if T_WL == 0 and T_RL == 0 and T_RH == 255 and T_GH == 255 and T_GL == 0:
        return image

    thresholded_img = threshold_color(image, thresholded_img, (0, 0, T_RL), (T_RH, T_RH, 255), RED)
    thresholded_img = threshold_color(image, thresholded_img, (0, T_GL, 0), (T_GH, 255, T_GH), GREEN)
    thresholded_img = threshold_color(image, thresholded_img, (T_WL, T_WL, T_WL), (255, 255, 255), WHITE)

    return thresholded_img

def filter_small_red(red_mask: np.ndarray, min_size: int) -> np.ndarray:
        if min_size == 1:
            return red_mask
        out_mask = np.zeros_like(red_mask)
        labels = measure.label(red_mask)
        for label in np.unique(labels):
            if label == 0:
                continue
            component = labels == label
            if np.sum(component) >= min_size:
                out_mask[component] = 1
        return out_mask

def fill_holes(bool_mask: np.ndarray)-> np.ndarray:
    mask = (bool_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 1, thickness=cv2.FILLED)
    return filled_mask.astype(bool)

def filter_color_noise(thresholed_image: np.ndarray, min_size)->np.ndarray:
    red_mask = np.all(thresholed_image == RED, axis=-1)
    min_red_mask = filter_small_red(red_mask, min_size=min_size)
    red_filled = fill_holes(min_red_mask)

    green_mask = np.all(thresholed_image == GREEN, axis=-1)
    lcc_green_mask = largest_cc(green_mask)
    green_filled = fill_holes(lcc_green_mask)

    white_mask = np.all(thresholed_image == WHITE, axis=-1)
    lcc_white_mask = largest_cc(white_mask)
    white_filled = fill_holes(lcc_white_mask)

    t_img = np.zeros_like(thresholed_image)
    t_img[red_filled] = RED
    t_img[green_filled] = GREEN
    t_img[white_filled] = WHITE 

    return t_img

def threshold_image2(image:np.ndarray, T_WL=190, T_RH=170, T_RL=120, 
                    T_GH=138, T_GL=140, min_size=5000)->np.ndarray:
    image = threshold_colors(image, T_WL, T_RH, T_RL, T_GH, T_GL)

    image = filter_color_noise(image, min_size)
    return image


def threshold_image(image:np.ndarray, T_WL=190, T_RH=170, T_RL=120, 
                    T_GH=138, T_GL=140, min_size=5000, dilatation=True)->np.ndarray:
    

    output_image = np.zeros_like(image)

    robot_mask = cv2.inRange(image, (T_WL, T_WL, T_WL), (255, 255, 255))
    contours, _ = cv2.findContours(robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        filled_mask = np.zeros_like(robot_mask)
        cv2.drawContours(filled_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
        output_image[filled_mask > 0] = [255, 255, 255] 

        if dilatation:
            _, radius = cv2.minEnclosingCircle(largest_contour)
            radius = int(np.ceil(radius)) 
            #kernel_size = 2 * radius + 1
            kernel_size = int(np.ceil(radius/2)) # test big kernel size on big scene later
            #kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    goal_mask = cv2.inRange(image, (0, T_GL, 0), (T_GH, 255, T_GH))
    contours, _ = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        filled_mask = np.zeros_like(goal_mask)
        cv2.drawContours(filled_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
        output_image[filled_mask > 0] = [0, 255, 0]  

    obs_mask = cv2.inRange(image, (0, 0, T_RL), (T_RH, T_RH, 255))
    min_red_mask = filter_small_red(obs_mask, min_size=min_size) 
    red_filled = fill_holes(min_red_mask) 
    if dilatation:
        if kernel is not None: 
            red_expanded = cv2.dilate(red_filled.astype(np.uint8), kernel, iterations=1)
        else:
            print("Warning: Robot not detected. Obstacles not expanded.")
            red_expanded = red_filled
        red_expanded = cv2.dilate(red_filled.astype(np.uint8), kernel, iterations=1)
        output_image[red_expanded > 0] = [0, 0, 255]
    else:
        output_image[red_filled > 0] = [0, 0, 255]

    return output_image

##################################################################################################


def get_dominant_object(block:np.ndarray, verbose: bool)->int:
    pixels = block.reshape(-1, 3)

    has_white = np.any(np.all(pixels == WHITE, axis=1))
    has_green = np.any(np.all(pixels == GREEN, axis=1))
    has_red = np.any(np.all(pixels == RED, axis=1))

    if has_white:
        if has_red and verbose:
            print("Collision start with obstacle")
        elif has_green and verbose:
            print("Collision start goal")
        return ROBOT # np.array([255, 255, 255], dtype=np.uint8)
    elif has_green:
        if has_red and verbose:
            print("Collision goal with obstacle")
        return GOAL # np.array([0, 255, 0], dtype=np.uint8)    
    elif has_red:
        return OBSTACLE # np.array([0, 0, 255], dtype=np.uint8)
    else:
        return BACKGROUND #np.array([0, 0, 0], dtype=np.uint8) 


def discretize_image(image:np.ndarray, grid_size: int, verbose: bool, full_output: bool):
    """
    Discretizes an OpenCV image using a grid of grid_size x grid_size cells.
    """
    height, width, _ = image.shape

    x_coords = np.linspace(0, width, num=grid_size+1, endpoint=True, dtype=int)
    y_coords = np.linspace(0, height, num=grid_size+1, endpoint=True, dtype=int)

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    if full_output:
        discretized_image = np.copy(image)

    for i in range(grid_size):
        for j in range(grid_size):
            start_x = x_coords[j]
            end_x = x_coords[j+1]
            start_y = y_coords[i]
            end_y = y_coords[i+1]

            block = image[start_y:end_y, start_x:end_x]

            dominant_object = get_dominant_object(block, verbose)

            grid[i, j] = dominant_object
            if full_output:
                discretized_image[start_y:end_y, start_x:end_x] = get_color(dominant_object)

    if full_output:
        return grid, discretized_image
    else:
        return grid
    
def image_to_grid(grid_image: np.ndarray) -> np.ndarray:
    grid = np.zeros((grid_image.shape[0], grid_image.shape[1]), dtype=np.int8)
    
    background_mask = np.all(grid_image == BLACK, axis=-1)
    start_mask = np.all(grid_image == WHITE, axis=-1)
    goal_mask = np.all(grid_image == GREEN, axis=-1)
    obstacle_mask = np.all(grid_image == RED, axis=-1)
    
    grid[background_mask] = BACKGROUND
    grid[start_mask] = ROBOT
    grid[goal_mask] = GOAL
    grid[obstacle_mask] = OBSTACLE
    
    return grid


def grid_to_image(grid: np.ndarray) -> np.ndarray:
    grid_image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    
    grid_image[grid == BACKGROUND] = BLACK # background/empty
    grid_image[grid == ROBOT] = WHITE # start/robot
    grid_image[grid == GOAL] = GREEN # goal
    grid_image[grid == OBSTACLE] = RED # obstacle   
    
    return grid_image

def get_grid(image: np.ndarray, grid_size=100, verbose=True, full_output=False)->np.ndarray:
    grid = discretize_image(image, grid_size, verbose, full_output)
    return grid


##################################################################################################

def get_centroids(grid:np.ndarray, _object):
    
    is_image = len(grid.shape) == 3 and grid.shape[2] == 3

    if isinstance(_object, str):
        if not is_image:
            object_code = get_object(_object)
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

def find_nose_corners(image: np.ndarray, sigma=5, t1=50, t2=150, threshold=100, minLineLength=100, maxLineGap=200)->np.ndarray:
    #edges = feature.canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sigma=sigma)
    temp = np.zeros_like(image)
    mask = np.all(image == WHITE, axis=-1)  
    temp[mask] = WHITE
    blurred_image = cv2.GaussianBlur(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY), (sigma, sigma), 1.4)
    edges = cv2.Canny(blurred_image, t1, t2)
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

def get_nose(image:np.array, sigma=5, t1=50, t2=150, threshold=25, minLineLength=20, maxLineGap=50):
    curve, bp1, bp2 = find_nose_corners(image, t1=t1, t2=t2, sigma=sigma, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
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
    return np.linalg.norm(np.array(a) - np.array(b), 2)

def a_star_search(map_grid, start, goal):

    start = tuple(start)
    goal = tuple(goal)

    #Initialize the open set as a priority queue and add the start node
    open_set = []
    heappush(open_set,(0+heuristic(start,goal),0,start))
    
    came_from = {}
    g_costs = {start: 0}
    explored = set()
    cost_map=-1*np.zeros_like(map_grid, dtype=np.int32)

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
            # Check if neighbor is within bounds and not an obstacle
            if (0 <= neighbor[0] < map_grid.shape[0]) and (0 <= neighbor[1] < map_grid.shape[1]) and (map_grid[neighbor]!=-1):
                
                # Calculate tentative_g_cost
                tentative_g_cost = g_costs[current_pos]+(map_grid[neighbor]) #cost is 1 y default on the map_grid
                #tentative_g_cost = g_costs[current_pos] + 1 # above was unstable after dilating the obstacles for some reason
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
        path[::-1]
        path = np.array(path).astype(np.int32)
        return path, explored, cost_map  # Return reversed path, explored cells and cost_map for visualization
    else:
        return None, None, explored
    
##################################################################################################
"""
def init(cam, sigma = 5, epsilon = 0.01, T_WL=190, T_RH=140, T_RL=120, T_GH=140, T_GL=120, min_size=5000, grid_size=200,
         threshold=25, minLineLength=20, maxLineGap=50):
    
    image = get_image_from_camera(cam) # camera calibration inside

    image = correct_perspective(image, sigma=sigma, epsilon=epsilon)

    image = threshold_image(image, T_WL, T_RH, T_RL, T_GH, T_GL, min_size)

    grid = get_grid(image, grid_size, verbose=True, full_output=False)

    grid_image = grid_to_image(grid)

    nose = get_nose(grid_image, sigma=sigma, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    c_obstacles = get_centroids(grid, "obstacle")
    c_robot = get_centroids(grid, "start")
    c_goal = get_centroids(grid, "goal")

    c_robot = c_robot.flatten()
    c_goal = c_goal.flatten()
    angle_rad, angle_deg = get_orientation(nose, c_robot)

    a_search_output = a_star_search(grid, c_robot, c_goal)

    return grid, c_robot, c_goal, c_obstacles, angle_rad, angle_deg, a_search_output
"""
def update_vision(cam, grid, sigma = 5, t1=50, t2=150, epsilon = 0.01, T_WL=190, t1_nose=50, t2_nose=150, threshold=25, minLineLength=20, maxLineGap=50):
    
    img = get_image_from_camera(cam)

    if img is not None:
        img = correct_perspective(img, sigma=sigma, t1=t1, t2=t2, epsilon=epsilon)
        threshold_image = np.zeros_like(img)
        robot_mask = cv2.inRange(img, (T_WL, T_WL, T_WL), (255, 255, 255))
        contours, _ = cv2.findContours(robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]
            filled_mask = np.zeros_like(robot_mask)
            cv2.drawContours(filled_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
            threshold_image[filled_mask > 0] = [255, 255, 255] 

            grid[grid == ROBOT] = 0
            filled_mask_resized = cv2.resize(filled_mask.astype(np.uint8), (grid.shape[1], grid.shape[0]), interpolation=cv2.INTER_NEAREST)
            grid[filled_mask_resized > 0] = ROBOT

            c_robot = get_centroids(grid, ROBOT)
            nose = get_nose(grid_to_image(grid), sigma=sigma, t1=t1_nose, t2=t2_nose, threshold= threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
            angle, angle_deg = get_orientation(nose, c_robot.flatten())

    return grid, c_robot, nose, angle, angle_deg