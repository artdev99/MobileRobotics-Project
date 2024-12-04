OBSTACLE_THRESHOLD = 1000
SPEED = 70

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
    braitenberg = [-10/300, -20/300, 30/300, 21/300, 11/300] #left to right
    v_mr, v_ml = SPEED, SPEED
    print("prox : ", prox_values)
    for i in range (len(prox_values)) :
        v_ml -= braitenberg[i]*prox_values[i]
        v_mr += braitenberg[i]*prox_values[i]
    print("braitenberg final speed : ", v_ml, v_mr)
    return v_ml, v_mr

    


    
        

    
    
    