import numpy as np

from utils.thymio import L_AXIS, normalize_angle

OBSTACLE_THRESHOLD = 800
SPEED = 150       #PWM
SPEED_LIMIT = 500 #PWM

def motion_control(thymio):

        k_alpha = 0.010*SPEED  #controls rotational velocity #85
        k_beta = 0      #damping term (to stabilize the robot's orientation when reaching the goal)

        x, y, theta, x_goal, y_goal = thymio.get_data_mm()

        delta_x = x_goal - x #[mm]
        delta_y = y_goal - y #[mm]

        delta_angle = normalize_angle(np.arctan2(delta_y, delta_x) - theta) #difference between the robot's orientation and the direction of the goal [rad]

        v = SPEED                                                  #translational velocity PWM
        omega = k_alpha*(delta_angle) - k_beta*(delta_angle+theta) #rotational velocity [rad/s]

        #Calculate motor speed
        v_ml = (v+omega*L_AXIS) #PWM
        v_mr = (v-omega*L_AXIS) #PWM

        return v_ml, v_mr

async def get_prox(node, client):
    await node.wait_for_variables({"prox.horizontal"})
    await client.sleep(0.05)
    return (list(node.v.prox.horizontal)[:-2])

def check_obstacle(prox_values):
    if max(prox_values) > OBSTACLE_THRESHOLD :
        return True
    else :
        return False        

def avoid_obstacle(prox_values): #left to right
    braitenberg = [-2/300, -10/300, 25/300, 11/300, 3/300] #left to right
    v_mr, v_ml = SPEED, SPEED
    for i in range (len(prox_values)) :
        v_ml -= braitenberg[i] * prox_values[i]
        v_mr += braitenberg[i] * prox_values[i]
    return v_ml, v_mr

async def set_motors(node, v_ml, v_mr): #v_l and v_r in PWM
    v_ml = limit_speed(v_ml)
    v_mr = limit_speed(v_mr)

    v_ml = int(v_ml)  #ensure integer type
    v_mr = int(v_mr)  #ensure integer type

    v_m = {
        "motor.left.target": [v_ml],
        "motor.right.target": [v_mr],
    }
    await node.set_variables(v_m)

def limit_speed(v):
    if(v > SPEED_LIMIT) :
        v = SPEED_LIMIT
    if(v < -SPEED_LIMIT) :
        v = -SPEED_LIMIT
    return v
        
