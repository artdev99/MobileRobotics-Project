import cv2
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt

GRID_L = 400  # [pixels]
GRID_W = 300  # [pixels]

ANGLE_THRESHOLD = np.radians(20)   # threshold under which changes of directions are ignored [rad]
STEP = 10                          # step (in number of cells) between each cell we study
COUNTER_THRESHOLD = 5              # max number of steps between keypoints

def discretize_image_eff(image):

    mask = np.all(image == [0, 0, 255], axis=2).astype(np.uint8)

    # Resize the mask to the desired grid size
    resized_mask = cv2.resize(
        mask, (GRID_L, GRID_W), interpolation=cv2.INTER_NEAREST
    ).astype(
        np.int32
    )  # cv2.INTER_AREA

    obstacle_grid = -2 * resized_mask + 1  # -1 obstacle, 1 rest

    return obstacle_grid

def heuristic(a, b):
    # Euclidian Distance
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def a_star_search(map_grid, start, goal, do_plot):

    # Initialize the open set as a priority queue and add the start node
    start = tuple(start.flatten())
    goal = tuple(goal.flatten())

    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start))

    came_from = {}
    g_costs = {start: 0}
    explored = set()
    cost_map = -1 * np.ones_like(map_grid, dtype=np.float64)

    while open_set:  # While the open set is not empty

        current_f_cost, current_g_cost, current_pos = heappop(open_set)
        # Add the current node to the explored set
        explored.add(current_pos)

        # Check if the goal has been reached
        if current_pos == goal:
            break
        # Get the neighbors of the current node 8 neighbors
        neighbors = [
            (current_pos[0], current_pos[1] + 1),
            (current_pos[0], current_pos[1] - 1),
            (current_pos[0] - 1, current_pos[1]),
            (current_pos[0] + 1, current_pos[1]),
            (current_pos[0] + 1, current_pos[1] + 1),
            (current_pos[0] - 1, current_pos[1] - 1),
            (current_pos[0] - 1, current_pos[1] + 1),
            (current_pos[0] + 1, current_pos[1] - 1),
        ]

        for neighbor in neighbors:
            if neighbor in explored:
                continue
            # Check if neighbor is within bounds and not an obstacle
            if (
                (0 <= neighbor[0] < map_grid.shape[0])
                and (0 <= neighbor[1] < map_grid.shape[1])
                and (map_grid[neighbor] > -1)
            ):

                # Calculate tentative_g_cost
                if (neighbor[0] == current_pos[0]) or (
                    neighbor[1] == current_pos[1]
                ):  # if goes straight
                    tentative_g_cost = g_costs[current_pos] + (
                        map_grid[neighbor]
                    )  # cost is 1 y default on the map_grid
                else:
                    tentative_g_cost = g_costs[current_pos] + (
                        map_grid[neighbor]
                    ) * np.sqrt(
                        2
                    )  # going diagonnaly is going further

                # If this path to neighbor is better than any previous one
                if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                    # Update came_from and g_costs
                    came_from[neighbor] = current_pos
                    g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic(neighbor, goal)
                    cost_map[neighbor] = f_cost
                    # Add neighbor to open set
                    heappush(open_set, (f_cost, tentative_g_cost, neighbor))

    # Reconstruct path
    if current_pos == goal:
        # Reconstruct the path
        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path = np.array(path[::-1]).T
        if do_plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(map_grid, cmap="Greys", origin="lower")

            # Plot explored nodes
            explored_nodes = np.array(list(explored))
            plt.plot(explored_nodes[:, 1], explored_nodes[:, 0], "y.", markersize=2)

            # Plot path
            if path is not None:
                plt.plot(path[1], path[0], "r-", linewidth=2)

            # Plot start and goal
            plt.plot(start[1], start[0], "go", markersize=10)  # start in green
            plt.plot(goal[1], goal[0], "bo", markersize=10)  # goal in blue

            plt.title("A* Path Finding")
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.gca().invert_yaxis()  # Optional: invert Y-axis to match matrix indexing
            plt.show()
        return (
            True,
            path,
            explored,
            cost_map,
        )  # Return reversed path, explored cells and cost_map for visualization
    else:
        return False, 0, 0, 0


def grid1_coord2grid2_coord(coord, grid1, grid2):
    coord_grid = np.copy(coord)
    if coord.ndim == 1:
        coord_grid[0] = coord[0] * (grid2.shape[0] - 1) / (grid1.shape[0] - 1)
        coord_grid[1] = coord[1] * (grid2.shape[1] - 1) / (grid1.shape[1] - 1)
    else:
        coord_grid[0, :] = coord[0, :] * (grid2.shape[0] - 1) / (grid1.shape[0] - 1)
        coord_grid[1, :] = coord[1, :] * (grid2.shape[1] - 1) / (grid1.shape[1] - 1)
    return np.int32(np.rint(coord_grid))


def find_rotation(dir_previous, dir_next):
    det = dir_previous[0] * dir_next[1] - dir_previous[1] * dir_next[0]
    dot_product = dir_previous[0] * dir_next[0] + dir_previous[1] * dir_next[1]
    theta = np.arctan2(det, dot_product)  # angle between the two directions [rad]
    return theta


def find_keypoints(path):

    path = path.T
    if len(path) < 3:
        return path

    keypoints = [path[0]] 
    counter = 1

    for i in range(STEP, len(path) - STEP, STEP):
        previous = path[i - STEP]  # previous cell
        current = path[i]  # current cell
        next = path[i + STEP]  # next cell

        # direction vectors
        dir_previous = (current[0] - previous[0], current[1] - previous[1])
        dir_next = (next[0] - current[0], next[1] - current[1])

        if (abs(find_rotation(dir_previous, dir_next)) > ANGLE_THRESHOLD):  # significant change of direction
            keypoints.append(current)
            # print("keypoint_angle : ", current)
            counter = 1
        elif (counter >= COUNTER_THRESHOLD):  # ensures there isn't too much space between keypoints (so we avoid accumulating ignored small changes of directions)
            keypoints.append(current)
            # print("keypoint_counter : ", current)
            counter = 1
        else:
            counter += 1

    keypoints.append(
        path[len(path) - 1]
    )  # last point of the path is the true final goal

    return keypoints
