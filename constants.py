import numpy as np

L_AXIS = 92                    #wheel axis length [mm]
SPEED_SCALING_FACTOR = 500/200 #Thymio cheat sheet : motors set at 500 -> translational velocity ≈ 200mm/s

def normalize_angle(angle): #restricts angle [rad] between -pi and pi
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
