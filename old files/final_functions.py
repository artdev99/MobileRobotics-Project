import cv2
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from skimage import measure


def full_detection_cnt_centroid(
    image: np.ndarray,
    thresh_obstacle,
    thresh_goal,
    min_size,
    corner_aruco_size_mm,
    corner_aruco_size_pix,
) -> np.ndarray:
    thresholded_img = np.zeros_like(image)
    Thymio_radius_mm = 70  # mm
    radius = Thymio_radius_mm * corner_aruco_size_pix / corner_aruco_size_mm
    # Find Obstacles
    obstacle_mask = 255 * np.ones(image.shape[:2], dtype=np.uint8)
    for i in range(thresh_obstacle.shape[0]):
        temp_mask = cv2.inRange(image, thresh_obstacle[i, :3], thresh_obstacle[i, 3:6])
        obstacle_mask = cv2.bitwise_and(obstacle_mask, temp_mask)
    obstacle_mask = filter_small_blobs(obstacle_mask, min_size=min_size)

    obstacle_mask, obstacle_cnt = fill_holes(obstacle_mask)

    distance = cv2.distanceTransform(~obstacle_mask, cv2.DIST_L2, 5)

    # Expand the obstacle by Thymio's radius
    expanded_obstacle_mask = (distance < radius) * 255

    expanded_obstacle_mask, obstacle_cnt_expnded = fill_holes(expanded_obstacle_mask)
    thresholded_img[expanded_obstacle_mask == 255] = [0, 0, 255]

    # Find Goal
    goal_mask = cv2.inRange(image, thresh_goal[:3], thresh_goal[3:6])

    goal_mask = filter_small_blobs(goal_mask, min_size=min_size)

    goal_mask, goal_cnt = fill_holes(goal_mask)

    thresholded_img[goal_mask == 255] = [0, 255, 0]
    # Goal Center:
    M = cv2.moments((goal_mask * 255).astype(np.uint8))
    Goal_x = int(M["m10"] / M["m00"])
    Goal_y = int(M["m01"] / M["m00"])
    Goal_center = np.array([Goal_x, Goal_y]).reshape(2, 1)

    return thresholded_img, obstacle_cnt, obstacle_cnt_expnded, goal_cnt, Goal_center


def fill_holes(bool_mask: np.ndarray) -> np.ndarray:
    mask = (bool_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    return filled_mask, contours


def filter_small_blobs(red_mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size == 1:
        return red_mask
    out_mask = np.zeros_like(red_mask)
    labels = measure.label(red_mask)
    for label in np.unique(labels):
        if label == 0:
            continue
        component = labels == label
        if np.sum(component) >= min_size:
            out_mask[component] = 255
    return out_mask


def order_points(corners):
    rect = np.zeros((4, 2), dtype="float32")

    _sum = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)

    rect[0] = corners[np.argmin(_sum)]
    rect[2] = corners[np.argmax(_sum)]
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    return rect


def compute_destination_size(ordered_corners):
    (top_left, top_right, bottom_right, bottom_left) = ordered_corners

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(top_left - bottom_left)
    height_right = np.linalg.norm(top_right - bottom_right)
    max_height = max(int(height_left), int(height_right))

    return max_width, max_height


def find_aruco_corners_size(image):

    # Initialize the detector parameters
    parameters = cv2.aruco.DetectorParameters()
    # Select the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

    # Detect the markers
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray_img, aruco_dict, parameters=parameters
    )
    if len(ids) < 4:
        raise ValueError("Not enough corners detected for perspective")
    inner_corners = []

    # Define the order of markers: top-left, bottom-left, bottom-right, top-right
    marker_order = [0, 1, 2, 10]
    aruco_corner = [
        2,
        1,
        0,
        3,
    ]  # Bottom-right, top-right, top-left, bottom-left of each aruco

    for marker_id, corner_pos in zip(marker_order, aruco_corner):

        idx = np.where(ids == marker_id)[0][0]
        # Get the inner corner
        inner_corners.append(corners[idx][0, corner_pos, :])
    size_aruco = []
    for i in range(4):
        side_lengths = [
            np.linalg.norm(corners[i][0, 0, :] - corners[i][0, 1, :]),  # Top side
            np.linalg.norm(corners[i][0, 1, :] - corners[i][0, 2, :]),  # Right side
            np.linalg.norm(corners[i][0, 2, :] - corners[i][0, 3, :]),  # Bottom side
            np.linalg.norm(corners[i][0, 3, :] - corners[i][0, 0, :]),  # Left side
        ]
        size_aruco.append(np.mean(side_lengths))
    return np.array(inner_corners), np.mean(size_aruco)


def discretize_image_eff(image, grid_size):

    mask = np.all(image == 255, axis=2).astype(np.uint8)

    # Resize the mask to the desired grid size
    resized_mask = cv2.resize(
        mask, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

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
            path,
            explored,
            cost_map,
        )  # Return reversed path, explored cells and cost_map for visualization
    else:
        raise ValueError("No path found")


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


def find_keypoints(path, ANGLE_THRESHOLD=np.radians(40), STEP=3, COUNTER_THRESHOLD=3):

    if len(path) < 3:
        return path

    keypoints = []
    keypoints.append(path[0])  # beginning of the path
    counter = 1

    for i in range(STEP, len(path) - STEP, STEP):
        previous = path[i - STEP]  # previous cell
        current = path[i]  # current cell
        next = path[i + STEP]  # next cell

        # direction vectors
        dir_previous = (current[0] - previous[0], current[1] - previous[1])
        dir_next = (next[0] - current[0], next[1] - current[1])

        if (
            abs(find_rotation(dir_previous, dir_next)) > ANGLE_THRESHOLD
        ):  # significant change of direction
            keypoints.append(current)
            counter = 1
        elif (
            counter >= COUNTER_THRESHOLD
        ):  # ensures there isn't too much space between keypoints (so we avoid accumulating ignored small changes of directions)
            keypoints.append(current)
            counter = 1
        else:
            counter += 1

    keypoints.append(
        path[len(path) - 1]
    )  # last point of the path is the true final goal

    return keypoints


def draw_on_image(camera, Thymio, path_img):
    image_cnt = camera.perspimage.copy()
    cv2.drawContours(image_cnt, camera.goal_cnt, -1, (0, 255, 0), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt, -1, (0, 0, 255), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt_expnded, -1, (0, 100, 255), 3)
    cv2.polylines(
        image_cnt,
        [path_img.T.reshape(-1, 1, 2)],
        isClosed=False,
        color=(255, 0, 0),
        thickness=3,
    )
    # if the line below oes crazy remove the .T TBD
    cv2.polylines(
        image_cnt,
        [Thymio.keypoints.T.reshape(-1, 1, 2)],
        isClosed=False,
        color=(255, 255, 151),
        thickness=2,
    )
    cv2.circle(image_cnt, camera.c_goal.flatten(), 10, (0, 255, 0), -1)

    if Thymio.Thymio_detected:
        Thymio_nose = (
            1.5
            * camera.size_aruco
            * np.array([np.cos(Thymio.xytheta_meas[2]), np.sin(Thymio.xytheta_meas[2])])
        )  # thymio is approx 1.5 aruco size
        Thymio_nose = Thymio_nose + Thymio.xytheta_meas[:2]
        cv2.arrowedLine(
            image_cnt,
            Thymio.xytheta_meas[:2].astype(int),
            Thymio_nose.astype(int),
            (255, 0, 255),
            2,
            tipLength=0.2,
        )
    # TBD add Thymio estimated + circle for variance (and two small arrows for angle variance?)
    cv2.imshow("Camera View", image_cnt)
    cv2.waitKey(1)
