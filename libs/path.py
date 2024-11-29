import math
import numpy as np
from heapq import heappush, heappop

def heuristic(a, b):
    #Euclidian Distance
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def a_star_search(map_grid, start, goal):

    #Initialize the open set as a priority queue and add the start node
    start=tuple(start.flatten())
    goal=tuple(goal.flatten())

    open_set = []
    heappush(open_set,(0+heuristic(start,goal),0,start))

    came_from = {}
    g_costs = {start: 0}
    explored = set()
    cost_map=-1*np.ones_like(map_grid,dtype=np.float64)

    while open_set:  # While the open set is not empty

        current_f_cost, current_g_cost, current_pos = heappop(open_set)
        # Add the current node to the explored set
        explored.add(current_pos)

        # Check if the goal has been reached
        if current_pos == goal:
            break
        # Get the neighbors of the current node 8 neighbors
        neighbors = [(current_pos[0],current_pos[1]+1),
                     (current_pos[0],current_pos[1]-1),
                     (current_pos[0]-1,current_pos[1]),
                     (current_pos[0]+1,current_pos[1]),
                     (current_pos[0]+1,current_pos[1]+1),
                     (current_pos[0]-1,current_pos[1]-1),
                     (current_pos[0]-1,current_pos[1]+1),
                     (current_pos[0]+1,current_pos[1]-1),

            ]

        for neighbor in neighbors:
            if neighbor in explored:
                continue
            # Check if neighbor is within bounds and not an obstacle
            if (0 <= neighbor[0] < map_grid.shape[0]) and (0 <= neighbor[1] < map_grid.shape[1]) and (map_grid[neighbor]!=-1):
                
                # Calculate tentative_g_cost
                if (neighbor[0]==current_pos[0]) or (neighbor[1]==current_pos[1]): #if goes straight
                    tentative_g_cost = g_costs[current_pos]+(map_grid[neighbor]) #cost is 1 y default on the map_grid
                else:
                    tentative_g_cost = g_costs[current_pos]+(map_grid[neighbor])*np.sqrt(2) #going diagonnaly is going further

                # If this path to neighbor is better than any previous one
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    # Update came_from and g_costs
                    came_from[neighbor] = current_pos
                    g_costs[neighbor] = tentative_g_cost
                    f_cost=tentative_g_cost+heuristic(neighbor,goal)
                    cost_map[neighbor]=f_cost
                    # Add neighbor to open set
                    heappush(open_set, (f_cost,tentative_g_cost,neighbor))
    
    # Reconstruct path
    if current_pos == goal:
        #Reconstruct the path
        path=[goal]
        while path[-1]!=start:
            path.append(came_from[path[-1]])
        return np.array(path[::-1]).T, explored, cost_map  # Return reversed path, explored cells and cost_map for visualization
    else:
        print("Error, no path")
        return None, explored, cost_map

def find_keypoints(path):
        
    def find_rotation(dir_previous,dir_next):
        det = dir_previous[0] * dir_next[1] - dir_previous[1] * dir_next[0]
        dot_product = dir_previous[0] * dir_next[0] + dir_previous[1] * dir_next[1]
        theta = math.atan2(det, dot_product) #oriented angle, in rad
        return theta

    #ANGLE_THRESHOLD = 0.15 #rad
    ANGLE_THRESHOLD = math.radians(40) #converts degrees to rad
    COUNTER_THRESHOLD = 3 #max number of steps between keypoints
    STEP = 3
        
    if len(path) < 3 :
        return path

    keypoints = [] 
    keypoints.append(path[0])
    counter = 1

    for i in range(STEP, len(path)-STEP, STEP): #beginning, max, step
        previous = path[i-STEP]
        current = path[i]
        next = path[i+STEP]

        #direction vectors
        dir_previous = (current[0] - previous[0], current[1] - previous[1])
        dir_next = (next[0] - current[0], next[1] - current[1])
        
        if (abs(find_rotation(dir_previous,dir_next)) > ANGLE_THRESHOLD): 
            keypoints.append(current)
            counter = 1
        elif (counter >= COUNTER_THRESHOLD): #ensures there isn't too much space between keypoints (so we avoid accumulating small changes of directions)
            keypoints.append(current)
            counter = 1
        else:
            counter += 1
            
    keypoints.append(path[len(path) - 1])  #last point of the path is the true final goal
    
    return keypoints