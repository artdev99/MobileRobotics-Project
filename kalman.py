import numpy as np
import time

from constants import *

class Kalman_class:

    def __init__(self, cam):
        self.Q = np.diag([15, 15, np.deg2rad(20)]) ** 2
        self.R = np.diag([5, 5, np.deg2rad(5)])** 2  # Measurement noise [0.0062, 0.0062, 0.0016] measureed in pix**2 (0.0586945)
        self.H = np.eye(3) 
        self.P = 10*self.R
        self.pixbymm = cam.pixbymm
        self.start_time=time.time()
        self.delta_t=0
        #self.v_var=151 # (v_var=var_L+var_R)

    def delta_time_update(self):
        self.delta_t=(time.time()-self.start_time)
        self.start_time=time.time()

    async def gather_data(self, node):
        v_L = []
        v_R = []
        for _ in range(10): #remove some variance
            await node.wait_for_variables({"motor.left.speed", "motor.right.speed"})
            v_L.append(node.v.motor.left.speed)
            v_R.append(node.v.motor.right.speed)
        v_L = np.mean(v_L)
        v_R = np.mean(v_R)

        return v_L, v_R

    def predict_state(self,thymio_xytheta_est,v_L,v_R):
        """
        Predict the next state
        """
        thymio_xytheta_est[:2]=thymio_xytheta_est[:2]/self.pixbymm #go in mm
        #print(f"Before scaling v_L {v_L} v_R {v_R}")
        
        v_L=v_L/SPEED_SCALING_FACTOR #go from pwm to mm/s
        v_R=v_R/SPEED_SCALING_FACTOR
        #print(f"AFTER scaling v_L {v_L} v_R {v_R}")
        theta = thymio_xytheta_est[2]
        
        # Compute linear and angular velocities
        v = (v_R + v_L) / 2
        omega = (v_L - v_R) /L_AXIS
        #print(f"73{v}")

        # Update state
        delta_theta = omega * self.delta_t
        theta_mid = theta + delta_theta / 2 #midpoint method (the robot is turning so we take avg angle)
        delta_x = v * np.cos(theta_mid) * self.delta_t
        delta_y = v * np.sin(theta_mid) * self.delta_t
        thymio_xytheta_est = thymio_xytheta_est + np.array([delta_x,delta_y,delta_theta])
        
        # Normalize angle to [-pi, pi]
        thymio_xytheta_est[2] = normalize_angle(thymio_xytheta_est[2])

        """
        Predict the next covariance matrix
        """
        # Compute Jacobian and covariance matrix
        G,Q = compute_G_Q(thymio_xytheta_est[2],v_L,v_R,L_AXIS,self.delta_t,self.Q)


        # Predict covariance
        self.P = G @ self.P @ G.T + Q
        thymio_xytheta_est[:2]=thymio_xytheta_est[:2]*self.pixbymm #go in pix

        return thymio_xytheta_est

    def update_state(self, thymio_xytheta_est, thymio_xytheta_meas):

        thymio_xytheta_est[:2] = thymio_xytheta_est[:2]/self.pixbymm #go in mm
        thymio_xytheta_meas = thymio_xytheta_meas/self.pixbymm #go in mm
        # Innovation
        
        y = thymio_xytheta_meas - self.H @ thymio_xytheta_est

        # Normalize angle difference to [-pi, pi]
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate
        thymio_xytheta_est = thymio_xytheta_est + K @ y
        # Normalize angle to [-pi, pi]
        thymio_xytheta_est[2] = normalize_angle(thymio_xytheta_est[2])

        # Update covariance estimate
        self.P = (np.eye(3) - K @ self.H) @ self.P
        
        
        thymio_xytheta_est[:2] = thymio_xytheta_est[:2]*self.pixbymm #go in pix
        thymio_xytheta_meas[:2] = thymio_xytheta_meas[:2]*self.pixbymm #go in pix
        
        return thymio_xytheta_est, thymio_xytheta_meas


def compute_G_Q(theta,v_L,v_R,wheel_base,dt,process_cov):
    """
    Compute the Jacobian G and covariance matrix Q
    """
    # Linear and angular velocities
    v = (v_R + v_L) / 2
    omega = (v_R - v_L) / wheel_base
    theta_mid = theta + omega * dt / 2 #midpoint method (the robot is turning)

    # Compute Jacobian
    G = np.array([
        [1, 0, -v * np.sin(theta_mid) * dt],
        [0, 1,  v * np.cos(theta_mid) * dt],
        [0, 0, 1]
    ])
    #Process cov
    Q=process_cov* dt

    return G,Q

