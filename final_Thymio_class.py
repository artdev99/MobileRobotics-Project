import cv2
import numpy as np
import time
########################
#Thymio class
########################
class Thymio_class:
    def __init__(self,Thymio_id,cam):

        self.Thymio_ID=Thymio_id
        self.Thymio_position_aruco(cam.persp_image)
        self.pixbymm=cam.pixbymm
        self.xytheta_est = self.xytheta_meas
        self.start_time=time.time()
        self.delta_t=0
        self.keypoints=None
        self.target_keypoint=None
        self.local_avoidance=False
        #Kalman
        self.kalman_wheel_base = 92 #mm
        self.kalman_process_cov = np.diag([1.0, 1.0, np.deg2rad(5)]) ** 2
        self.kalman_measurement_cov = np.diag([1, 1, 0.0016])  # Measurement noise [0.0062, 0.0062, 0.0016] measureed in pix**2 (0.0586945)
        self.kalman_P=100*self.kalman_measurement_cov
        self.v_var=151 # (v_var=var_L+var_R)



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
#Kalman
    def kalman_predict_state(self,v_L,v_R):
        """
        Predict the next state
        """
        self.xytheta_est[:2]=self.xytheta_est[:2]/self.pixbymm #go in mm

        theta =self.xytheta_est[2]

        # Compute linear and angular velocities
        v = (v_R + v_L) / 2
        omega = (v_R - v_L) /self.kalman_wheel_base

        # Update state
        delta_theta = omega * self.delta_t
        theta_mid = theta + delta_theta / 2 #midpoint method (the robot is turning so nwe take avg angle)
        delta_x = v * np.cos(theta_mid) * self.delta_t
        delta_y = v * np.sin(theta_mid) * self.delta_t

        self.xytheta_est = self.xytheta_est + np.array([delta_x,delta_y,delta_theta])
        
        # Normalize angle to [-pi, pi]
        self.xytheta_est[2] = (self.xytheta_est[2] + np.pi) % (2 * np.pi) - np.pi

        """
        Predict the next covariance matrix
        """
        # Compute Jacobian and covariance matrix
        F,Q = compute_F_Q(self.xytheta_est[2],v_L,v_R,self.kalman_wheel_base,self.delta_t,self.kalman_process_cov,self.v_L_var,self.v_R_var)


        # Predict covariance
        self.P = F @ self.P @ F.T + Q
        
        self.xytheta_est[:2]=self.xytheta_est[:2]*self.pixbymm #go in pix

    def kalman_update_state(self):

        self.xytheta_est[:2]=self.xytheta_est[:2]/self.pixbymm #go in mm
        
        H = np.eye(3) #We measure the states directly

        # Innovation
        y = self.xytheta_meas/self.pixbymm - H @ self.xytheta_est

        # Normalize angle difference to [-pi, pi]
        y[2, 0] = (y[2, 0] + np.pi) % (2 * np.pi) - np.pi

        # Innovation covariance
        S = H @ self.kalman_P @ H.T + self.kalman_measurement_cov

        # Kalman gain
        K = self.kalman_P @ H.T @ np.linalg.inv(S)

        # Update state estimate
        self.xytheta_est = self.xytheta_est + K @ y

        # Normalize angle to [-pi, pi]
        self.xytheta_est[2] = (self.xytheta_est[2] + np.pi) % (2 * np.pi) - np.pi

        # Update covariance estimate
        self.kalman_P = (np.eye(3) - K @ H) @ self.kalman_P

        self.xytheta_est[:2]=self.xytheta_est[:2]*self.pixbymm #go in pix


def compute_F_Q(theta,v_L,v_R,wheel_base,dt,process_cov,v_var):
    """
    Compute the Jacobian F and covariance matrix Q
    """
    # Linear and angular velocities
    v = (v_R + v_L) / 2
    omega = (v_R - v_L) / wheel_base
    theta_mid = theta + omega * dt / 2 #midpoint method (the robot is turning)

    # Compute Jacobian
    F = np.array([
        [1, 0, -v * np.sin(theta_mid) * dt],
        [0, 1,  v * np.cos(theta_mid) * dt],
        [0, 0, 1]
    ])

    omega_var = v_var / wheel_base** 2

    # Process noise covariance
    Q = np.array([
        [v_var * dt ** 2, 0, 0],
        [0, v_var * dt ** 2, 0],
        [0, 0, omega_var * dt ** 2]
    ]) + process_cov

    return F,Q

