import numpy as np
import cv2
from skimage import measure

########################
#Camera
########################
class camera_class:
    def __init__(self,camera_index=1,corner_aruco_id=[0, 1, 2, 10],corner_aruco_size=70,min_size=5000, thresh_obstacle=np.array([0,0,120,0,0,140]), thresh_goal=np.array([0,120,0,0,140,0])):

        self.cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            self.cam.release()
            raise IOError("Camera could not be opened")
        
        #Warmup
        for _ in range(50):  # Camera needs to do lighting and white balance adjustments
            ret, _ = self.cam.read()
            if not ret:
                self.cam.release()
                raise IOError("Failed to capture frame during warmup")

        #We get the original image of the camer:
        self.get_image(False) #We get the image without distorsion correction
        
        #We correct the perspective and get the perspective matrices to not have to find the corners at each iteration
        self.corner_aruco_id=corner_aruco_id
        self.correct_perspective_aruco(get_matrix=True)
        self.pixbymm=self.size_aruco/corner_aruco_size
        #We get the image with expanded obstacles and all the contours
        self.thresholded_image,self.obstacle_cnt, self.obstacle_cnt_expnded, self.goal_cnt,self.goal_center= full_detection_cnt_centroid(self.persp_image, thresh_obstacle, thresh_goal, min_size, self.size_aruco, corner_aruco_size)
        


    def get_image(self,distortion=False, alpha=1):
        ret, self.image = self.cam.read()
        if not ret:
            self.cam.release()
            raise IOError("Failed to get image")

        #if distortion:
        #    self.image = self.correct_camera_distortion(self.image, alpha)

    


    def correct_perspective_aruco(self,get_matrix=False) -> np.ndarray:
        if get_matrix:
            
            corners, self.size_aruco= find_aruco_corners_size(self.image)

            ordered_corners = order_points(corners)
            self.max_width_perspective, self.max_height_perspective = compute_destination_size(ordered_corners)
            destination_corners = np.array([
                [0, 0],
                [self.max_width_perspective - 1, 0],
                [self.max_width_perspective - 1, self.max_height_perspective - 1],
                [0, self.max_height_perspective - 1]], dtype="float32")
            self.M = cv2.getPerspectiveTransform(ordered_corners, destination_corners)

        self.persp_image = cv2.warpPerspective(self.image,self.M, (self.max_width_perspective, self.max_height_perspective), flags=cv2.INTER_LINEAR)
        

###Functions
def full_detection_cnt_centroid(image: np.ndarray, thresh_obstacle, thresh_goal,min_size, corner_aruco_size_mm,corner_aruco_size_pix) -> np.ndarray:
    thresholded_img = np.zeros_like(image)
    Thymio_radius_mm=70 #mm
    radius=0.38*Thymio_radius_mm*corner_aruco_size_pix/corner_aruco_size_mm
    # Find Obstacles
    obstacle_mask=255*np.ones(image.shape[:2], dtype=np.uint8)
    for i in range(thresh_obstacle.shape[0]):
        temp_mask = cv2.inRange(image, thresh_obstacle[i,:3], thresh_obstacle[i,3:6])
        obstacle_mask = cv2.bitwise_and(obstacle_mask, temp_mask)
    obstacle_mask = filter_small_blobs(obstacle_mask, min_size=min_size)

    obstacle_mask, obstacle_cnt = fill_holes(obstacle_mask)

    distance = cv2.distanceTransform(~obstacle_mask, cv2.DIST_L2, 5)

    # Expand the obstacle by Thymio's radius
    expanded_obstacle_mask = (distance < radius) * 255

    expanded_obstacle_mask, obstacle_cnt_expnded = fill_holes(expanded_obstacle_mask)
    thresholded_img[expanded_obstacle_mask==255] = [0, 0, 255]

    # Find Goal
    goal_mask = cv2.inRange(image, thresh_goal[:3], thresh_goal[3:6])

    contours, _ = cv2.findContours(goal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    goal_cnt = max(contours, key=cv2.contourArea)
    goal_mask =np.zeros_like(goal_mask)
    cv2.drawContours(goal_mask, goal_cnt, -1, 255, thickness=cv2.FILLED)
    thresholded_img[goal_mask==255] = [0, 255, 0]
    #Goal Center:
    M = cv2.moments((goal_mask*255).astype(np.uint8))
    Goal_x = int(M["m10"] / M["m00"])
    Goal_y = int(M["m01"] / M["m00"])
    Goal_center=np.array([Goal_x,Goal_y]).reshape(2,1)

    return thresholded_img,obstacle_cnt, obstacle_cnt_expnded, goal_cnt, Goal_center


def fill_holes(bool_mask: np.ndarray)-> np.ndarray:
    mask = (bool_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask,contours


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


def find_aruco_corners_size(image):

    # Initialize the detector parameters
    parameters = cv2.aruco.DetectorParameters()
    # Select the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    
    # Detect the markers
    gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)
    if len(ids)<4:
        raise ValueError("Not enough corners detected for perspective")
    inner_corners = []

    # Define the order of markers: top-left, bottom-left, bottom-right, top-right
    marker_order = [0, 1, 2, 10]
    aruco_corner = [2, 1, 0, 3]  # Bottom-right, top-right, top-left, bottom-left of each aruco
    aruco_corner = [0, 3, 2, 1]  # Bottom-right, top-right, top-left, bottom-left of each aruco

    for marker_id, corner_pos in zip(marker_order, aruco_corner):

        idx = np.where(ids == marker_id)[0][0]
        # Get the inner corner
        inner_corners.append(corners[idx][0,corner_pos,:])
    size_aruco=[]
    for i in range(4):
        side_lengths = [
            np.linalg.norm(corners[i][0,0,:] - corners[i][0,1,:]),  # Top side
            np.linalg.norm(corners[i][0,1,:] - corners[i][0,2,:]),  # Right side
            np.linalg.norm(corners[i][0,2,:] - corners[i][0,3,:]),  # Bottom side
            np.linalg.norm(corners[i][0,3,:] - corners[i][0,0,:])   # Left side
        ]
        size_aruco.append(np.mean(side_lengths))
    return np.array(inner_corners), np.mean(size_aruco)



def draw_on_image(camera,Thymio,path_img):
    image_cnt=camera.persp_image.copy()
    cv2.drawContours(image_cnt, camera.goal_cnt, -1, (0,255,0), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt, -1, (0,0,255), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt_expnded, -1, (0,100,255), 3)
    cv2.polylines(image_cnt, [path_img.T.reshape(-1,1,2)], isClosed=False, color=(255, 0, 0), thickness=3)
    cv2.circle(image_cnt,camera.goal_center.flatten(), 10, (0,255,0), -1)
    for i in range(len(Thymio.keypoints)):
        cv2.circle(image_cnt, Thymio.keypoints[i], 10, (200, 240, 190), -1)
    cv2.circle(image_cnt, Thymio.target_keypoint, 10, (0, 255, 255), -1)

    radius=1.5*camera.size_aruco
    if Thymio.Thymio_detected:
        Thymio_nose=radius*np.array([np.cos(Thymio.xytheta_meas[2]),np.sin(Thymio.xytheta_meas[2])]) #thymio is approx 1.5 aruco size
        Thymio_nose=Thymio_nose+Thymio.xytheta_meas[:2]
        cv2.arrowedLine(image_cnt, Thymio.xytheta_meas[:2].astype(int),Thymio_nose.astype(int) , (255, 0, 255), 2, tipLength=0.2)
    

    
    #Kalman:
    #2sigma-confidence Position (95%)

    cv2.circle(image_cnt, Thymio.xytheta_est[:2].astype(int), (2*np.sqrt(Thymio.kalman_P[2,2])*Thymio.pixbymm).astype(int), (0, 255, 255), 2)
    #Angle:
    # sigma-confidence arc (95%)
    start_angle = np.degrees(Thymio.xytheta_est[2]) - np.degrees(2*np.sqrt(Thymio.kalman_P[2,2]))  # Start of the arc
    end_angle = np.degrees(Thymio.xytheta_est[2]) + np.degrees(2*np.sqrt(Thymio.kalman_P[2,2]))    # End of the arc

    cv2.ellipse(image_cnt, Thymio.xytheta_est[:2].astype(int), (radius.astype(int), radius.astype(int)), 0, start_angle, end_angle, (255, 0, 127), 2)
    

    cv2.imshow('Camera View', image_cnt)
    cv2.waitKey(1)

def draw_history(camera,Thymio,path_img):
    image_cnt=camera.persp_image.copy()
    cv2.drawContours(image_cnt, camera.goal_cnt, -1, (0,255,0), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt, -1, (0,0,255), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt_expnded, -1, (0,100,255), 3)
    cv2.polylines(image_cnt, [path_img.T.reshape(-1,1,2)], isClosed=False, color=(255, 0, 0), thickness=3)
    cv2.circle(image_cnt,camera.goal_center.flatten(), 10, (0,255,0), -1)

    #Draw history
    cv2.polylines(image_cnt, [Thymio.xytheta_meas_hist[:,:2].reshape(-1,1,2)], isClosed=False, color=(255, 0, 255), thickness=2)
    cv2.polylines(image_cnt, [Thymio.xytheta_est_hist[:,:2].reshape(-1,1,2)], isClosed=False, color=(0, 255, 255), thickness=2)
    cv2.imshow('History', image_cnt)
    cv2.waitKey(1)
