import math

def motion_control(x,y,theta,x_goal,y_goal):

    R_WHEEL = 4.3 #radius of the wheels
    L = 9.2 #wheel axis length
    DISTANCE_THRESHOLD = 1 #what is the unity of x and x_goal ?
    ANGLE_THRESHOLD = 0.1 #radians
    SPEED_LIMIT = 500

    def normalize_angle(angle): #restricts angle between -pi and pi
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def goal_reached(distance_to_goal,delta_angle):
        return ((distance_to_goal<DISTANCE_THRESHOLD) and (abs(delta_angle)<DISTANCE_THRESHOLD))

    def limit_speed(v):
        if(v>SPEED_LIMIT) :
            v=SPEED_LIMIT
        if(v<-SPEED_LIMIT) :
            v=-SPEED_LIMIT
        return v
    
    #To do : Tune k by testing
    k_rho = 1.0 # To control linear velocity
    k_alpha = 3.0 # To control angular velocity 
    k_beta = -0.5 # Damping term (to stabilize the robot's orientation when reaching the goal)

    delta_x = x_goal - x
    delta_y = y_goal - y

    distance_to_goal =math.sqrt( (delta_x)**2 + (delta_y)**2 )
    delta_angle = normalize_angle(math.atan2(delta_y, delta_x) - theta) #difference between the robot's orientation and the direction of the goal
    #print("IN: d=", distance_to_goal, " angle=",delta_angle)

    v = k_rho*distance_to_goal #linear velocity
    omega = k_alpha*(delta_angle) - k_beta*(delta_angle+theta) #angular velocity
    
    #Calculate motor speed
    w_ml = (v+omega*L)//R_WHEEL #rad/s 
    w_mr = (v-omega*L)//R_WHEEL
    #print("IN: w_ml, w_mr : ", w_ml, w_mr)
    
    v_ml = w_ml*200//math.pi #PWM
    v_mr = w_mr*200//math.pi
    #print("IN: v_ml, v_mr : ", v_ml, v_mr)

    if(goal_reached(distance_to_goal,delta_angle)):
        v_ml=0
        v_mr=0
        print("goal reached !")
    
    v_ml=v_ml//80
    v_mr=v_mr//80

    v_ml = limit_speed(v_ml)
    v_mr = limit_speed(v_mr)
    print("IN: v_ml, v_mr : ", v_ml, v_mr)
    
    v_ml = int(v_ml)  # Ensure integer type
    v_mr = int(v_mr)  # Ensure integer type

    v_m = {
        "motor.left.target": [v_ml],
        "motor.right.target": [v_mr],
    }
    
    return v_m
