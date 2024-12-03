import cv2
import numpy as np
import time

THYMIO_ARUCO_ID = 9

class Thymio:
    def __init__(self, thymio_id, cam):
        self.thymio_id = thymio_id
        self.pixbymm = cam.pixbymm
        self.thymio_detected = False
        self.xytheta_meas = np.zeros(3)  # Initialize measurement vector
        self.xytheta_est = np.zeros(3)   # Initialize estimated state
        self.start_time = time.time()
        self.delta_t = 0
        self.keypoints = None
        self.target_keypoint = None
        self.local_avoidance = False

        # Kalman Filter parameters
        self.kalman_wheel_base = 92  # mm
        self.kalman_process_cov = np.diag([1.0, 1.0, np.deg2rad(5)]) ** 2
        self.kalman_measurement_cov = np.diag([1.0, 1.0, 0.0016])
        self.kalman_P = 100 * self.kalman_measurement_cov.copy()
        self.v_var = 151  # Variance of the velocity

        # Initialize P matrix
        self.P = self.kalman_P.copy()

        # Attempt to detect Thymio position
        self.update_thymio_position(cam.persp_image)
        if self.thymio_detected:
            self.xytheta_est = self.xytheta_meas.copy()