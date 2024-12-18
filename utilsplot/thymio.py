import cv2
import numpy as np
import time

THYMIO_ID = 9
L_AXIS = 92                    #wheel axis length [mm]
SPEED_SCALING_FACTOR = 1/0.38  #measured
#Thymio cheat sheet : motors set at 500 -> translational velocity ≈ 200mm/s, SPEED_SCALING_FACTOR = 1/0.4
KIDNAPPING_THRESHOLD = 500     #for prox.ground.delta
DISTANCE_THRESH = 60  #[mm]
 
########################
# Thymio class
########################
class Thymio_class:
    def __init__(self, cam):

        self.Thymio_ID = THYMIO_ID
        self.Thymio_position_aruco(cam.persp_image)
        self.pixbymm = cam.pixbymm
        self.xytheta_est = self.xytheta_meas
        self.start_time = time.time()
        self.delta_t = 0
        self.keypoints = None
        self.target_keypoint = None
        self.local_avoidance = False
        self.xytheta_meas_hist = np.empty((0, 3))
        self.xytheta_est_hist = np.empty((0, 3))
        # Kalman
        """self.kalman_Q = np.diag([7.5, 7.5, np.deg2rad(5)]) ** 2
        self.kalman_R = (
            np.diag([5, 5, np.deg2rad(1)]) ** 2
        )  # Measurement noise [0.0062, 0.0062, 0.0016] measureed in pix**2 (0.0586945)
        self.kalman_H = np.eye(3)"""
        
        self.kalman_Q = np.diag([23, 23, 0.401])
        self.kalman_R = (
            np.diag([5, 5, np.deg2rad(2)])
        )  # Measurement noise [0.0062, 0.0062, 0.0016] measureed in pix**2 (0.0586945)
        self.kalman_H = np.eye(3)
        self.kalman_P =  self.kalman_R
        self.kalman_Q_ctrl=np.diag([31, 29])
        # self.v_var=151 # (v_var=var_L+var_R)

    def Thymio_position_aruco(self, img):

        # Initialize the detector parameters
        parameters = cv2.aruco.DetectorParameters()
        # Select the ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

        # Detect the markers
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cv2.__version__ == "4.10.0":
            _aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = _aruco_detector.detectMarkers(gray_img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray_img, aruco_dict, parameters=parameters
            )

        if (ids is None) or (self.Thymio_ID not in ids):
            self.Thymio_detected = False
        else:
            idx = np.where(ids == self.Thymio_ID)[0][0]  # Thymio's aruco ID is 9
            aruco_corners = np.array(corners[idx][0, :, :])

            # Thymio's center:
            Thymio_x, Thymio_y = aruco_corners.mean(axis=0)

            # Thymio's angle
            top_edge = aruco_corners[1, :] - aruco_corners[0, :]
            bottom_edge = aruco_corners[2, :] - aruco_corners[3, :]
            angle = np.mean(
                [
                    np.arctan2(bottom_edge[1], bottom_edge[0]),
                    np.arctan2(top_edge[1], top_edge[0]),
                ]
            )

            self.xytheta_meas = np.array([Thymio_x, Thymio_y, angle])
            self.Thymio_detected = True
        
    def get_data_mm(self):
        x_mm = ((self.xytheta_est.flatten())[0])/self.pixbymm
        y_mm = ((self.xytheta_est.flatten())[1])/self.pixbymm
        theta_rad = self.xytheta_est.flatten()[2]
        x_goal_mm=((self.target_keypoint.flatten())[0])/self.pixbymm
        y_goal_mm=((self.target_keypoint.flatten())[1])/self.pixbymm
        return x_mm, y_mm, theta_rad, x_goal_mm, y_goal_mm
    
    def reached_goal(self):
        x, y, _, x_goal, y_goal = self.get_data_mm()
        delta_x = x_goal - x #[mm]
        delta_y = y_goal - y #[mm]
        distance_to_goal = np.sqrt( (delta_x)**2 + (delta_y)**2 ) #[mm]
        return (distance_to_goal < DISTANCE_THRESH)
    
# Kalman

    def delta_time_update(self):
        self.delta_t = time.time() - self.start_time
        self.start_time = time.time()

    def kalman_predict_state(self, v_L, v_R):
        """
        Predict the next state
        """
        self.xytheta_est[:2] = self.xytheta_est[:2] / self.pixbymm  # go in mm

        v_L = v_L / SPEED_SCALING_FACTOR  # go from pwm to mm/s
        v_R = v_R / SPEED_SCALING_FACTOR
        theta = self.xytheta_est[2]

        # Compute linear and angular velocities
        v = (v_R + v_L) / 2
        omega = (v_L - v_R) / L_AXIS

        # Update state
        delta_theta = omega * self.delta_t
        theta_mid = theta + delta_theta / 2 # midpoint method (the robot is turning so we take avg angle)
        delta_x = v * np.cos(theta_mid) * self.delta_t
        delta_y = v * np.sin(theta_mid) * self.delta_t
        self.xytheta_est = self.xytheta_est + np.array([delta_x, delta_y, delta_theta])

        # Normalize angle to [-pi, pi]
        self.xytheta_est[2] = normalize_angle(self.xytheta_est[2])

        """
        Predict the next covariance matrix
        """
        # Compute Jacobian and covariance matrix
        G, Q = compute_G_Q(
            self.xytheta_est[2],
            v_L,
            v_R,
            L_AXIS,
            self.delta_t,
            self.kalman_Q,
        )

        # Predict covariance
        G_u=compute_Gu(theta, v_R, v_L, L_AXIS, self.delta_t)
        self.kalman_P = G @ self.kalman_P @ G.T  + G_u @ self.kalman_Q_ctrl @ G_u.T #+ Q 
        self.xytheta_est[:2] = self.xytheta_est[:2] * self.pixbymm  # go in pix

    def kalman_update_state(self):

        self.xytheta_est[:2] = self.xytheta_est[:2] / self.pixbymm  # go in mm
        self.xytheta_meas[:2] = self.xytheta_meas[:2] / self.pixbymm  # go in mm

        # Innovation
        y = self.xytheta_meas - self.kalman_H @ self.xytheta_est

        # Normalize angle difference to [-pi, pi]
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi

        # Innovation covariance
        S = self.kalman_H @ self.kalman_P @ self.kalman_H.T + self.kalman_R

        # Kalman gain
        K = self.kalman_P @ self.kalman_H.T @ np.linalg.inv(S)

        # Update state estimate
        self.xytheta_est = self.xytheta_est + K @ y
        # Normalize angle to [-pi, pi]
        self.xytheta_est[2] = normalize_angle(self.xytheta_est[2])

        # Update covariance estimate
        self.kalman_P = (np.eye(3) - K @ self.kalman_H) @ self.kalman_P

        self.xytheta_est[:2] = self.xytheta_est[:2] * self.pixbymm  # go in pix
        self.xytheta_meas[:2] = self.xytheta_meas[:2] * self.pixbymm  # go in pix

def compute_G_Q(theta, v_L, v_R, wheel_base, dt, process_cov):
    """
    Compute the Jacobian G and covariance matrix Q
    """
    # Linear and angular velocities
    v = (v_R + v_L) / 2
    omega = (v_R - v_L) / wheel_base
    theta_mid = theta + omega * dt / 2  # midpoint method (the robot is turning)

    # Compute Jacobian
    G = np.array(
        [
            [1, 0, -v * np.sin(theta_mid) * dt],
            [0, 1, v * np.cos(theta_mid) * dt],
            [0, 0, 1],
        ]
    )
    # Process cov
    Q = process_cov * dt

    return G, Q

def compute_Gu(theta, v_R, v_L, wheel_base, dt):
    """
    Compute the Jacobian Gu (derivative of motion equations w.r.t. control inputs)
    """

    # Precompute common terms to be a bit faster :)
    v = 0.5 * (v_L + v_R) 
    omega = (v_L - v_R) / wheel_base 
    theta_mid = theta + 0.5 * omega * dt  # Midpoint angle

    cos_theta = np.cos(theta_mid)
    sin_theta = np.sin(theta_mid)

    delta_t_sq = dt ** 2
    common_factor = (v * delta_t_sq) / (4.0 * wheel_base)

    # Compute partial derivatives
    G_u=np.empty((3,2))
    G_u[0,0] = 0.5 * cos_theta - common_factor * sin_theta
    G_u[0,1] = 0.5 * cos_theta + common_factor * sin_theta

    G_u[1,0] = 0.5 * sin_theta + common_factor * cos_theta
    G_u[1,1] = 0.5 * sin_theta - common_factor * cos_theta

    G_u[2,0] = dt / wheel_base
    G_u[2,1] = -dt / wheel_base
    return G_u




async def gather_data(node):
    v_L = []
    v_R = []
    for _ in range(10): #remove some variance
        await node.wait_for_variables({"motor.left.speed", "motor.right.speed"})
        v_L.append(node.v.motor.left.speed)
        v_R.append(node.v.motor.right.speed)
    v_L = np.mean(v_L)
    v_R = np.mean(v_R)

    return v_L, v_R

async def check_kidnapping(node):
    # await node.wait_for_variables({"acc"})
    # if(abs(node.v.acc[2])<KIDNAPPING_THRESHOLD):
    #     return True
    # else:
    #     return False
    
    await node.wait_for_variables({"prox.ground.delta"})
    if(min(node.v.prox.ground.delta)<KIDNAPPING_THRESHOLD):
        return True
    else:
        return False
    
def normalize_angle(angle): #restricts angle [rad] between -pi and pi
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


