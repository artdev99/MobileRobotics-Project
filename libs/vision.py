import cv2
import numpy as np
from skimage import feature, measure
from scipy.spatial import distance
import os

def camera(): 
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW) #Specify DirectShow for faster connection
    if not cam.isOpened(): 
        print("Camera could not be opened") 
        cam.release()
        exit()

    #Create folder for the Calibration Images
    folder_name = "Chessboard_Calib_imgs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    img_count=0
    while img_count<20:
        ret, frame = cam.read()
        if not ret:
            print("Frame could not be read for some reason")
            cam.release()
            exit()

        cv2.imshow(f"Take image {img_count+1} by pressing s",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            filepath = os.path.join(folder_name, f"Calib_img{img_count}.png")
            cv2.imwrite(filepath,frame)
            img_count+=1
            cv2.destroyAllWindows()
        

    cv2.destroyAllWindows()
    cam.release()

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

def correct_distortion(image: np.ndarray, sigma=5, epsilon=0.01, eps_security=True, verbose=False) -> np.ndarray:
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


def threshold_colors(image: np.ndarray, T_WL: int, T_RH: int, T_RL: int, T_GH: int, T_GL: int) -> np.ndarray:
    thresholded_img = np.zeros_like(image)
    
    if T_WL == 0 and T_RL == 0 and T_RH == 255 and T_GH == 255 and T_GL == 0:
        return image

    red_mask = cv2.inRange(image, (0, 0, T_RL), (T_RH, T_RH, 255))
    thresholded_img[red_mask > 0] = [0, 0, 255]

    green_mask = cv2.inRange(image, (0, T_GL, 0), (T_GH, 255, T_GH))
    thresholded_img[green_mask > 0] = [0, 255, 0]

    white_mask = cv2.inRange(image, (T_WL, T_WL, T_WL), (255, 255, 255))
    thresholded_img[white_mask > 0] = [255, 255, 255]

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
    red_mask = np.all(thresholed_image == [0, 0, 255], axis=-1)
    min_red_mask = filter_small_red(red_mask, min_size=min_size)
    red_filled = fill_holes(min_red_mask)

    green_mask = np.all(thresholed_image == [0, 255, 0], axis=-1)
    lcc_green_mask = largest_cc(green_mask)
    green_filled = fill_holes(lcc_green_mask)

    white_mask = np.all(thresholed_image == [255, 255, 255], axis=-1)
    lcc_white_mask = largest_cc(white_mask)
    white_filled = fill_holes(lcc_white_mask)

    t_img = np.zeros_like(thresholed_image)
    t_img[red_filled] = [0, 0, 255]
    t_img[green_filled] = [0, 255, 0]
    t_img[white_filled] = [255, 255, 255] 

    return t_img

def threshold_image(image:np.ndarray, T_WL=190, T_RH=170, T_RL=120, 
                    T_GH=138, T_GL=140, min_size=5000)->np.ndarray:
    image = threshold_colors(image, T_WL, T_RH, T_RL, T_GH, T_GL)
    image = filter_color_noise(image, min_size)
    return image


##################################################################################################


def get_dominant_color(block:np.ndarray, verbose: bool)->np.ndarray:
    pixels = block.reshape(-1, 3)

    has_white = np.any(np.all(pixels == [255, 255, 255], axis=1))
    has_green = np.any(np.all(pixels == [0, 255, 0], axis=1))
    has_red = np.any(np.all(pixels == [0, 0, 255], axis=1))

    if has_white:
        if has_red and verbose:
            print("Collision start with obstacle")
        elif has_green and verbose:
            print("Collision start goal")
        return np.array([255, 255, 255], dtype=np.uint8)
    elif has_green:
        if has_red and verbose:
            print("Collision goal with obstacle")
        return np.array([0, 255, 0], dtype=np.uint8)    
    elif has_red:
        return np.array([0, 0, 255], dtype=np.uint8)
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
    
    grid[background_mask] = 0
    grid[start_mask] = 1
    grid[goal_mask] = 2
    grid[obstacle_mask] = -1
    
    return grid


def grid_to_image(grid: np.ndarray) -> np.ndarray:
    grid_image = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    
    grid_image[grid == 0] = [0, 0, 0] # background/empty
    grid_image[grid == 1] = [255, 255, 255] # start/robot
    grid_image[grid == 2] = [0, 255, 0] # goal
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
        "start": 1,
        "robot": 1,
        "goal": 2,
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

def get_nose(image:np.array, sigma=5):
    curve, bp1, bp2 = find_nose_corners(image, sigma=sigma)
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