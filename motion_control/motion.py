import math

#TO DO : What are the units of x and y (the parameters) ? If not in cm -> fix units

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
        return ((distance_to_goal<DISTANCE_THRESHOLD) and (abs(delta_angle)<ANGLE_THRESHOLD))

    def limit_speed(v):
        if(v>SPEED_LIMIT) :
            v=SPEED_LIMIT
        if(v<-SPEED_LIMIT) :
            v=-SPEED_LIMIT
        return v
    
    #To do : Tune k by testing
    k_rho = 1.0     #controls translational velocity
    k_alpha = 3.0   #controls rotational velocity 
    k_beta = -0.5   #damping term (to stabilize the robot's orientation when reaching the goal)

    delta_x = x_goal - x #[cm]
    delta_y = y_goal - y #[cm]

    distance_to_goal =math.sqrt( (delta_x)**2 + (delta_y)**2 )          #[cm]
    delta_angle = normalize_angle(math.atan2(delta_y, delta_x) - theta) #difference between the robot's orientation and the direction of the goal [rad]
    #print("IN: d=", distance_to_goal, " angle=",delta_angle)

    v = k_rho*distance_to_goal                                  #translational velocity [cm/s]
    omega = k_alpha*(delta_angle) - k_beta*(delta_angle+theta)  #rotational velocity [rad/s]
    
    #Calculate motor speed
    w_ml = (v+omega*L_AXIS)/R_WHEEL #[rad/s]
    w_mr = (v-omega*L_AXIS)/R_WHEEL #[rad/s]
    #print("IN: w_ml, w_mr : ", w_ml, w_mr)
    
    v_ml = w_ml*SCALING_FACTOR #PWM
    v_mr = w_mr*SCALING_FACTOR #PWM
    #print("IN: v_ml, v_mr : ", v_ml, v_mr)

    if(goal_reached(distance_to_goal,delta_angle)):
        v_ml=0
        v_mr=0
        print("goal reached !")
    
    v_ml = limit_speed(v_ml)
    v_mr = limit_speed(v_mr)
    print("IN: v_ml, v_mr : ", v_ml, v_mr)
    
    v_ml = int(v_ml)  #ensure integer type
    v_mr = int(v_mr)  #ensure integer type

    v_m = {
        "motor.left.target": [v_ml],
        "motor.right.target": [v_mr],
    }
    
    return v_m
