import numpy as np

from constants import *

SPEED = 50 #[mm/s] 

def motion_control(thymio):

        k_alpha = 0.35  #controls rotational velocity 
        k_beta = 0      #damping term (to stabilize the robot's orientation when reaching the goal)

        x, y, theta, x_goal, y_goal = thymio.get_data_mm()

        delta_x = x_goal - x #[mm]
        delta_y = y_goal - y #[mm]

        delta_angle = normalize_angle(np.arctan2(delta_y, delta_x) - theta) #difference between the robot's orientation and the direction of the goal [rad]

        v = SPEED                                                   #translational velocity [mm/s]
        omega = k_alpha*(delta_angle) - k_beta*(delta_angle+theta)  #rotational velocity [rad/s]

        #Calculate motor speed
        v_ml = (v+omega*L_AXIS)*SPEED_SCALING_FACTOR #PWM
        v_mr = (v-omega*L_AXIS)*SPEED_SCALING_FACTOR #PWM

        return v_ml, v_mr