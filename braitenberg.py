OBSTACLE_THRESHOLD = 1000
SPEED = 70
SPEED_LIMIT = 500 

async def get_prox(node, client):
    await node.wait_for_variables({"prox.horizontal"})
    await client.sleep(0.05)
    return (list(node.v.prox.horizontal)[:-2])

def check_obstacle(prox_values):
    if max(prox_values) > OBSTACLE_THRESHOLD :
        return True
    else :
        return False
    
def limit_speed(v):
    if(v>SPEED_LIMIT) :
        v=SPEED_LIMIT
    if(v<-SPEED_LIMIT) :
        v=-SPEED_LIMIT
    return v          

def avoid_obstacle(prox_values): #left to right
    braitenberg = [-10/300, -20/300, 30/300, 21/300, 11/300] #left, center, right
    v_mr, v_ml = SPEED, SPEED
    print("prox : ", prox_values)
    for i in range (len(prox_values)) :
        v_ml -= braitenberg[i]*prox_values[i]
        v_mr += braitenberg[i]*prox_values[i]
        
    v_ml = limit_speed(v_ml)
    v_mr = limit_speed(v_mr)
    v_ml = int(v_ml)  #ensure integer type
    v_mr = int(v_mr)  #ensure integer type
    v_m = {
        "motor.left.target": [v_ml],
        "motor.right.target": [v_mr],
    }
    print("braitenberg final speed : ", v_m)
    return v_m

    


    
        

    
    
    