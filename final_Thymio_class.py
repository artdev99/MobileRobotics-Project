import cv2
import numpy as np
import time
########################
#Thymio class
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