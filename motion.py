import numpy as np

def pixel_to_mm(value_pixel, pixbymm):
    value_mm = value_pixel/pixbymm
    return value_mm   #[mm]

def adjust_units(Thymio_xytheta, c_goal, pixbymm):
    x_mm = pixel_to_mm((Thymio_xytheta.flatten())[0], pixbymm)
    y_mm = pixel_to_mm((Thymio_xytheta.flatten())[1], pixbymm)
    theta_rad = Thymio_xytheta.flatten()[2]
    x_goal_mm=pixel_to_mm((c_goal.flatten())[0], pixbymm)
    y_goal_mm=pixel_to_mm((c_goal.flatten())[1], pixbymm)
    #x_goal_mm= 405/pixbymm
    #y_goal_mm= 234/pixbymm
    return x_mm, y_mm, theta_rad, x_goal_mm, y_goal_mm

def distance_to_goal(x,y,x_goal,y_goal):
    delta_x = x_goal - x #[mm]
    delta_y = y_goal - y #[mm]
    distance_to_goal =np.sqrt( (delta_x)**2 + (delta_y)**2 ) #[mm]
    return distance_to_goal

def motion_control(x,y,theta,x_goal,y_goal):

    R_WHEEL = 43                      #wheels radius [mm]
    L_AXIS = 92                       #wheel axis length [mm]
    DISTANCE_THRESHOLD = 30           #margin to consider goal reached [mm]
    ANGLE_THRESHOLD = 0.1             #margin to consider goal reached [rad]
    SPEED_LIMIT = 500                 #PWM
    SCALING_FACTOR = 500/(200/R_WHEEL) #Thymio cheat sheet : motors set at 500 -> translational velocity ≈ 200mm/s

    def normalize_angle(angle): #restricts angle [rad] between -pi and pi
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def goal_reached(distance_to_goal,delta_angle):
        return (distance_to_goal<DISTANCE_THRESHOLD)
        #return ((distance_to_goal<DISTANCE_THRESHOLD) and (abs(delta_angle)<ANGLE_THRESHOLD))

    def limit_speed(v):
        if(v>SPEED_LIMIT) :
            v=SPEED_LIMIT
        if(v<-SPEED_LIMIT) :
            v=-SPEED_LIMIT
        return v
    
    #To do : Tune k by testing
    k_rho = 0.1     #controls translational velocity
    k_alpha = 0.4   #controls rotational velocity 
    k_beta = 0      #damping term (to stabilize the robot's orientation when reaching the goal)

    delta_x = x_goal - x #[mm]
    delta_y = y_goal - y #[mm]

    distance_to_goal =np.sqrt( (delta_x)**2 + (delta_y)**2 )          #[mm]
    delta_angle = normalize_angle(np.arctan2(delta_y, delta_x) - theta) #difference between the robot's orientation and the direction of the goal [rad]
    #print("IN: d=", distance_to_goal, " angle=",math.degrees(delta_angle))

    v = k_rho*distance_to_goal                                  #translational velocity [mm/s]
    v = 70                                                      #translational velocity [mm/s]
    omega = k_alpha*(delta_angle) - k_beta*(delta_angle+theta)  #rotational velocity [rad/s]
    #print("omega = ", omega)
    
    #Calculate motor speed
    w_ml = (v+omega*L_AXIS)/R_WHEEL #[rad/s]
    w_mr = (v-omega*L_AXIS)/R_WHEEL #[rad/s]
    #print("after omega: w_ml, w_mr : ", w_ml, w_mr)
    
    v_ml = w_ml*SCALING_FACTOR #PWM
    v_mr = w_mr*SCALING_FACTOR #PWM
    #print("before: v_ml, v_mr : ", v_ml, v_mr)

    #if(goal_reached(distance_to_goal,delta_angle)):
        #v_ml=0
        #v_mr=0
        #print("goal reached !")
    
    #print("v_ml : ", v_ml, "v_mr : ", v_mr)
    
    v_ml = limit_speed(v_ml)
    v_mr = limit_speed(v_mr)
    
    v_ml = int(v_ml)  #ensure integer type
    v_mr = int(v_mr)  #ensure integer type

    v_m = {
        "motor.left.target": [v_ml],
        "motor.right.target": [v_mr],
    }
    
    return v_m
