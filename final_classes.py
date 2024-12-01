###########################################################
#
# This file contains the Camera and Thymio classes
#
###########################################################
import numpy as np
import cv2
import time
from final_functions import *

########################
#Camera
########################
class camera_class:
    def __init__(self,camera_index=1,corner_aruco_id=[0, 1, 2, 10],min_size=5000, thresh_obstacle=np.array([0,0,120,0,0,140]), thresh_goal=np.array([0,120,0,0,140,0])):

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
        #We get the image with expanded obstacles and all the contours
        self.thresholded_image,self.obstacle_cnt, self.obstacle_cnt_expnded, self.goal_cnt,self.goal_center= full_detection_cnt_centroid(self.persp_image, thresh_obstacle, thresh_goal, min_size)
        


    def get_image(self,distortion=False, alpha=1):
        ret, self.image = self.cam.read()
        if not ret:
            self.cam.release()
            raise IOError("Failed to get image")

        #if distortion:
        #    self.image = self.correct_camera_distortion(self.image, alpha)

    


    def correct_perspective_aruco(self,get_matrix=False) -> np.ndarray:
        self.size_aruco=-1
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
        


########################
#Thymio
########################
class Thymio_class:
    def __init__(self,Thymio_id,image):

        self.Thymio_id=Thymio_id
        self.Thymio_position_aruco(image)
        self.xytheta_est = self.xytheta_meas
        self.speed=np.zeros((1,2))
        self.start_time=time.time()
        self.delta_t=0
        self.keypoints=None
        self.target_keypoint=None
        self.local_avoidance=False


    def Thymio_position_aruco(self,img):

        # Initialize the detector parameters
        parameters = cv2.aruco.DetectorParameters()
        # Select the ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        
        # Detect the markers
        gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)

        if (ids is None) or (self.Thymio_ID not in ids):
            self.Thymio_detected=False
        else:
            idx = np.where(ids == self.Thymio_ID)[0][0] #Thymio's aruco ID is 10
            aruco_corners=np.array(corners[idx][0,:,:])

            #Thymio's center:
            Thymio_x,Thymio_y=aruco_corners.mean(axis=0)

            #Thymio's angle
            top_edge=aruco_corners[1,:]-aruco_corners[0,:]
            bottom_edge=aruco_corners[2,:]-aruco_corners[3,:]
            angle = np.mean([np.arctan2(bottom_edge[1], bottom_edge[0]), 
                            np.arctan2(top_edge[1], top_edge[0])])

            self.xytheta_meas = np.array([Thymio_x,Thymio_y,angle])
            self.Thymio_detected=True

    def delta_time_update(self):
        self.delta_t=time.time()-self.start_time
        self.start_time=time.time()



