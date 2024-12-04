from constants import *

OBSTACLE_THRESHOLD = 1000
SPEED = 70

async def get_prox(node, client):
    await node.wait_for_variables({"prox.horizontal"})
    await client.sleep(0.05)
    return (list(node.v.prox.horizontal)[:-2])

async def check_obstacle(prox_values, node):
    #await node.wait_for_variables({"acc"})
    #if(abs(node.v.acc[2])<KIDNAPPING_THRESHOLD): #doesn't care about obstacles if being kidnapped
    if max(prox_values) > OBSTACLE_THRESHOLD :
        return True
    else :
        return False        

def avoid_obstacle(prox_values): #left to right
    braitenberg = [-10/300, -20/300, 30/300, 21/300, 11/300] #left to right
    v_mr, v_ml = SPEED, SPEED
    for i in range (len(prox_values)) :
        v_ml -= braitenberg[i]*prox_values[i]
        v_mr += braitenberg[i]*prox_values[i]
    return v_ml, v_mr

    


    
        

    
    
    