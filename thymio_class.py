import cv2
import numpy as np

from constants import *
       
########################
#Thymio class
########################
class Thymio_class:
    def __init__(self,Thymio_id,cam):

        self.Thymio_ID=Thymio_id
        self.Thymio_position_aruco(cam.persp_image)
        self.pixbymm=cam.pixbymm
        self.xytheta_est = self.xytheta_meas
        self.keypoints=None
        self.target_keypoint=None
        self.xytheta_meas_hist=np.empty((0,3))
        self.xytheta_est_hist=np.empty((0,3))

    def Thymio_position_aruco(self,img):

        # Initialize the detector parameters
        parameters = cv2.aruco.DetectorParameters()
        # Select the ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        
        # Detect the markers
        gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cv2.__version__ == "4.10.0":
            _aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = _aruco_detector.detectMarkers(gray_img)
        else:
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
        
    def get_data_mm(self):
        x_mm = ((self.xytheta_est.flatten())[0])/self.pixbymm
        y_mm = ((self.xytheta_est.flatten())[1])/self.pixbymm
        theta_rad = self.xytheta_est.flatten()[2]
        x_goal_mm=((self.target_keypoint.flatten())[0])/self.pixbymm
        y_goal_mm=((self.target_keypoint.flatten())[1])/self.pixbymm
        return x_mm, y_mm, theta_rad, x_goal_mm, y_goal_mm
    
    def distance_to_goal(self):
        x, y, _, x_goal, y_goal = self.get_data_mm()
        delta_x = x_goal - x #[mm]
        delta_y = y_goal - y #[mm]
        distance_to_goal = np.sqrt( (delta_x)**2 + (delta_y)**2 ) #[mm]
        return distance_to_goal
