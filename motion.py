import math

def pixel_to_cm(value_pixel, pixbymm):
    value_cm = (value_pixel/pixbymm)/10
    return value_cm   #[cm]

def adjust_units(Thymio_xytheta, c_goal, pixbymm):
    x_cm = pixel_to_cm((Thymio_xytheta.flatten())[0], pixbymm)
    y_cm = pixel_to_cm((Thymio_xytheta.flatten())[1], pixbymm)
    theta_rad = Thymio_xytheta.flatten()[2]
    #x_goal_cm=pixel_to_cm((c_goal.flatten())[0], pixbymm)
    #y_goal_cm=pixel_to_cm((c_goal.flatten())[1], pixbymm)
    x_goal_cm= (405/pixbymm)/10
    y_goal_cm= (234/pixbymm)/10
    return x_cm, y_cm, theta_rad, x_goal_cm, y_goal_cm

def motion_control(x,y,theta,x_goal,y_goal):

    R_WHEEL = 4.3                     #wheels radius [cm]
    L_AXIS = 9.2                      #wheel axis length [cm]
    DISTANCE_THRESHOLD = 3            #margin to consider goal reached [cm]
    ANGLE_THRESHOLD = 0.1             #margin to consider goal reached [rad]
    SPEED_LIMIT = 500                 #PWM
    SCALING_FACTOR = 500/(20/R_WHEEL) #Thymio cheat sheet : motors set at 500 -> translational velocity ≈ 20cm/s

    def normalize_angle(angle): #restricts angle [rad] between -pi and pi
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
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
    k_rho = 1.0     #controls translational velocity
    k_alpha = 4.0   #controls rotational velocity 
    k_beta = 0      #damping term (to stabilize the robot's orientation when reaching the goal)

    delta_x = x_goal - x #[cm]
    delta_y = y_goal - y #[cm]

    distance_to_goal =math.sqrt( (delta_x)**2 + (delta_y)**2 )          #[cm]
    delta_angle = normalize_angle(math.atan2(delta_y, delta_x) - theta) #difference between the robot's orientation and the direction of the goal [rad]
    print("IN: d=", distance_to_goal, " angle=",math.degrees(delta_angle))

    v = k_rho*distance_to_goal                                  #translational velocity [cm/s]
    omega = k_alpha*(delta_angle) - k_beta*(delta_angle+theta)  #rotational velocity [rad/s]
    print("omega = ", omega)
    
    #Calculate motor speed
    w_ml = (v+omega*L_AXIS)/R_WHEEL #[rad/s]
    w_mr = (v-omega*L_AXIS)/R_WHEEL #[rad/s]
    print("after omega: w_ml, w_mr : ", w_ml, w_mr)
    
    v_ml = w_ml*SCALING_FACTOR #PWM
    v_mr = w_mr*SCALING_FACTOR #PWM
    #print("before: v_ml, v_mr : ", v_ml, v_mr)

    if(goal_reached(distance_to_goal,delta_angle)):
        v_ml=0
        v_mr=0
        print("goal reached !")
    
    v_ml = v_ml/10
    v_mr = v_mr/10
    print("v_ml : ", v_ml, "v_mr : ", v_mr)
    
    v_ml = limit_speed(v_ml)
    v_mr = limit_speed(v_mr)
    
    v_ml = int(v_ml)  #ensure integer type
    v_mr = int(v_mr)  #ensure integer type

    v_m = {
        "motor.left.target": [v_ml],
        "motor.right.target": [v_mr],
    }
    
    return v_m
