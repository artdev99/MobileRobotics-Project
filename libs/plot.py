import numpy as np
import cv2
import matplotlib.pyplot as plt 
from ipywidgets import interact, FloatSlider, IntSlider
from skimage import feature
from libs.vision import largest_cc, find_corners, filter_small_red, find_nose_corners, get_midpoint, get_slope_intercept, get_centroids, get_orientation



def show_cv2_image(img: np.ndarray, fig_size=(12,12), color="RGB", _axis=True, _title=None, _cmap=None):
    plt.figure(figsize=fig_size)
    if color == "BGR":
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=_cmap) # BGR -> RGB
    else:
        if color != "RGB":
            print(f"color {color} not recognized, applying default RGB")
        plt.imshow(img, cmap=_cmap)
        
    if _title is not None:
        plt.title(_title)
    if not _axis:
        plt.axis('off')
    
    plt.show()


def show_distortion(image: np.ndarray, sigma_init=5, epsilon_int=0.0025, epsilon_max=0.003, circle_size=10, border_size=3):
    def update(sigma, epsilon):
        edges = feature.canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), sigma=sigma)
        mask = largest_cc(edges)
        corners = find_corners(mask, epsilon=epsilon, eps_security=False, verbose=True)

        img_corners = image.copy()
        cv2.drawContours(img_corners, [corners.reshape(-1, 2)], -1, (255, 255, 0), border_size)
        for corner in corners:
            cv2.circle(img_corners, tuple(corner), circle_size, (255, 0, 255), -1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Distortion Correction (sigma={sigma:.2f}, epsilon={epsilon:.4f})", fontsize=16)
        
        # Input image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Input Image")
        axes[0].axis("on")
        
        # Edges
        axes[1].imshow(edges, cmap='gray')
        axes[1].set_title("Edges")
        axes[1].axis("on")
        
        # Corners on the original image
        axes[2].imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Corners")
        axes[2].axis("on")
        
        plt.show()
    
    interact(
        update, 
        sigma=FloatSlider(value=sigma_init, min=1, max=10, step=0.1, description='Sigma', readout_format='.2f'),
        epsilon=FloatSlider(value=epsilon_int, min=0.0001, max=epsilon_max, step=0.0001, description='Epsilon', readout_format='.4f')
    )

def show_thresholds(image: np.ndarray, T_WL_init=190, T_RH_init=140, T_RL_init=120, T_GH_init=140, T_GL_init=120, min_size_init=1000):
    height, width, _ = image.shape
    
    def update(T_WL, T_RH, T_RL, T_GH, T_GL, min_size, x_coord):
        red_mask = cv2.inRange(image, (0, 0, T_RL), (T_RH, T_RH, 255))
        red_threshold = np.zeros_like(image)
        red_threshold[red_mask > 0] = [0, 0, 255]

        green_mask = cv2.inRange(image, (0, T_GL, 0), (T_GH, 255, T_GH))
        green_threshold = np.zeros_like(image)
        green_threshold[green_mask > 0] = [0, 255, 0]

        white_mask = cv2.inRange(image, (T_WL, T_WL, T_WL), (255, 255, 255))
        white_threshold = np.zeros_like(image)
        white_threshold[white_mask > 0] = [255, 255, 255]
        
        combined_thresholds = np.zeros_like(image)
        combined_thresholds[red_mask > 0] = [0, 0, 255]
        combined_thresholds[green_mask > 0] = [0, 255, 0]
        combined_thresholds[white_mask > 0] = [255, 255, 255]

        lcc_white_mask = largest_cc(white_mask)
        lcc_green_mask = largest_cc(green_mask)
        filtered_red_mask = filter_small_red(np.all(combined_thresholds == [0, 0, 255], axis=-1), min_size)
        
        filtered_white = np.zeros_like(image)
        filtered_white[lcc_white_mask] = [255, 255, 255]
        
        filtered_green = np.zeros_like(image)
        filtered_green[lcc_green_mask] = [0, 255, 0]
        
        filtered_red = np.zeros_like(image)
        filtered_red[filtered_red_mask > 0] = [0, 0, 255]

        x_coord = min(max(0, x_coord), width - 1)
        red_intensity = image[:, x_coord, 2]
        green_intensity = image[:, x_coord, 1]
        blue_intensity = image[:, x_coord, 0]

        fig = plt.figure(figsize=(18, 18))
        fig.suptitle(f"Threshold Visualization (T_WL={T_WL}, T_RH={T_RH}, T_RL={T_RL}, "
                     f"T_GH={T_GH}, T_GL={T_GL}, min_size={min_size}, x={x_coord})", fontsize=16)

        ax1 = fig.add_subplot(3, 3, 1)
        ax1.imshow(cv2.cvtColor(white_threshold, cv2.COLOR_BGR2RGB))
        ax1.set_title("White Threshold")
        ax1.axis("off")
        
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.imshow(cv2.cvtColor(red_threshold, cv2.COLOR_BGR2RGB))
        ax2.set_title("Red Threshold")
        ax2.axis("off")
        
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.imshow(cv2.cvtColor(green_threshold, cv2.COLOR_BGR2RGB))
        ax3.set_title("Green Threshold")
        ax3.axis("off")
        
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax4.set_title("Input Image")
        ax4.axis("on")

        gs = fig.add_gridspec(3, 3)
        gs_bottom = gs[1, 1].subgridspec(3, 1)
        ax_r = fig.add_subplot(gs_bottom[0])
        ax_r.plot(red_intensity, color='red')
        ax_r.set_xlim([0, height])
        ax_r.set_ylabel("Red Intensity")
        ax_g = fig.add_subplot(gs_bottom[1])
        ax_g.plot(green_intensity, color='green')
        ax_g.set_xlim([0, height])
        ax_g.set_ylabel("Green Intensity")
        ax_b = fig.add_subplot(gs_bottom[2])
        ax_b.plot(blue_intensity, color='blue')
        ax_b.set_xlim([0, height])
        ax_b.set_xlabel("Y Coordinate")
        ax_b.set_ylabel("Blue Intensity")

        ax5 = fig.add_subplot(3, 3, 6)
        ax5.imshow(cv2.cvtColor(combined_thresholds, cv2.COLOR_BGR2RGB))
        ax5.set_title("Combined Thresholds")
        ax5.axis("on")

        ax6 = fig.add_subplot(3, 3, 7)
        ax6.imshow(cv2.cvtColor(filtered_white, cv2.COLOR_BGR2RGB))
        ax6.set_title("LCC White")
        ax6.axis("off")

        ax7 = fig.add_subplot(3, 3, 8)
        ax7.imshow(cv2.cvtColor(filtered_red, cv2.COLOR_BGR2RGB))
        ax7.set_title("Min Size Red")
        ax7.axis("off")
        
        ax8 = fig.add_subplot(3, 3, 9)
        ax8.imshow(cv2.cvtColor(filtered_green, cv2.COLOR_BGR2RGB))
        ax8.set_title("LCC Green")
        ax8.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    interact(
        update, 
        T_WL=IntSlider(value=T_WL_init, min=0, max=255, step=1, description='T_WL'),
        T_RH=IntSlider(value=T_RH_init, min=0, max=255, step=1, description='T_RH'),
        T_RL=IntSlider(value=T_RL_init, min=0, max=255, step=1, description='T_RL'),
        T_GH=IntSlider(value=T_GH_init, min=0, max=255, step=1, description='T_GH'),
        T_GL=IntSlider(value=T_GL_init, min=0, max=255, step=1, description='T_GL'),
        min_size=IntSlider(value=min_size_init, min=1, max=3000, step=100, description='Min Size'),
        x_coord=IntSlider(value=width // 2, min=0, max=width - 1, step=1, description='X Coord')
    )

def show_grid(grid: np.ndarray, fig_size=(6,6), c_obs=None, c_robot=None, c_goal=None):
    height, width = grid.shape
    grid_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Checkerboard pattern
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                grid_image[i:(i + 1), j:(j + 1)] = [230, 230, 230]
            else:
                grid_image[i:(i + 1), j:(j + 1)] = [200, 200, 200]

    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:  # Start/robot
                grid_image[i:(i + 1), j:(j + 1)] = [255, 255, 255]
            elif grid[i, j] == 2:  # Goal
                grid_image[i:(i + 1), j:(j + 1)] = [0, 255, 0]
            elif grid[i, j] == -1:  # Obstacle
                grid_image[i:(i + 1), j:(j + 1)] = [255, 0, 0]

    if c_obs is not None:
        for c in c_obs:
            grid_image[c[1], c[0]] = [0, 0, 255]  # Blue for obstacles

    if c_robot is not None:
        for c in c_robot:
            grid_image[c[1], c[0]] = [255, 165, 0]  # Orange for start/robot

    if c_goal is not None:
        for c in c_goal:
            grid_image[c[1], c[0]] = [0, 80, 0]  # Dark green for goal

    plt.figure(figsize=fig_size)
    plt.imshow(grid_image)
    if c_obs is not None and c_robot is not None and c_goal is not None:
        plt.title(f"Grid {width} x {height} with centroids")
    else: 
        plt.title(f"Grid {width} x {height}")
    plt.axis("on")
    plt.show()


def show_nose(image:np.array, sigma_init=5, threshold_init=50, minLineLength_init=20, maxLineGap_init=50, circleSize_init=10):
    def update(sigma, threshold, minLineLength, maxLineGap, circleSize):
        img = image.copy()


        curve, bp1, bp2 = find_nose_corners(img, sigma=sigma, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
        mid = get_midpoint(bp1, bp2)
        centroid = get_centroids(img, [255,255,255])
        slope, intercept = get_slope_intercept(mid, centroid.flatten())
        if slope is None:
            x = mid[0]
            for y in range(curve.shape[0]):
                if curve[y, x] > 0:
                    return np.array([x, y], dtype=int)
        
        contour_points = np.argwhere(curve > 0)
        
        nose = None

        # for iteractive hough
        edges = feature.canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma=sigma)
        mask = largest_cc(edges).astype(np.uint8) * 255
        lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
        lines_mask = np.zeros_like(edges).astype(np.uint8) * 255
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_mask, (x1, y1), (x2, y2), 255, thickness=1)
        curved_border = mask - lines_mask
        curved_border = largest_cc(curved_border).astype(np.uint8) * 255

        for point in contour_points:
            x, y = point[1], point[0]
            if abs(y - (slope * x + intercept)) < 1:
                nose = np.array([x, y], dtype=int)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Nose Visualization (Sigma={sigma}, Threshold={threshold}, MinLineLength={minLineLength}, "
                     f"MaxLineGap={maxLineGap})", fontsize=16)

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("on")

        cv2.circle(img, tuple(bp1), circleSize, (255, 255, 0), -1) 
        cv2.circle(img, tuple(bp2), circleSize, (255, 255, 0), -1) 
        cv2.circle(img, tuple(mid), circleSize, (255, 0, 255), -1) 
        cv2.circle(img, tuple(centroid.flatten()), circleSize, (0, 165, 255), -1)

        if nose is not None:
            cv2.circle(img, tuple(nose), circleSize, (0, 255, 255), -1)

        axes[1].imshow(curved_border, cmap="gray")
        axes[1].set_title("Hough")
        axes[1].axis("on")

        axes[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Identified Points")
        axes[2].axis("on")

        plt.show()

        _, angle_deg = get_orientation(nose, centroid.flatten())

        print(f"Nose: {nose}")
        print(f"Orientation: {angle_deg}°")
        print(f"Centroid: {centroid.flatten()}")
        print(f"Border point 1: {bp1}")
        print(f"Border point 2: {bp2}")
        print(f"Intersection: {mid}")

    interact(
        update,
        sigma=FloatSlider(value=sigma_init, min=1, max=10, step=0.1, description='Sigma', readout_format='.2f'),
        threshold=IntSlider(value=threshold_init, min=1, max=200, step=1, description='Threshold'),
        minLineLength=IntSlider(value=minLineLength_init, min=1, max=200, step=1, description='Min Line Length'),
        maxLineGap=IntSlider(value=maxLineGap_init, min=1, max=200, step=1, description='Max Line Gap'),
        circleSize=IntSlider(value=circleSize_init, min=1, max=10, step=1, description='Circle Size')
    )

# image threshold=100, minLineLength=120, maxLineGap=200
# grid_image threshold=20, minLineLength=20, maxLineGap=50
