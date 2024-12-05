import numpy as np
import cv2
import matplotlib.pyplot as plt
import ipywidgets
from ipywidgets import IntSlider, VBox, Output
from IPython.display import display
import os

def draw_on_image(camera, Thymio, path_img):
    image_cnt = camera.persp_image.copy()
    cv2.drawContours(image_cnt, camera.goal_cnt, -1, (0, 255, 0), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt, -1, (0, 0, 255), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt_expnded, -1, (0, 100, 255), 3)
    
    if (path_img is not None): 
        cv2.polylines(
            image_cnt,
            [path_img.T.reshape(-1, 1, 2)],
            isClosed=False,
            color=(255, 0, 0),
            thickness=3,
        )
    cv2.circle(image_cnt, camera.goal_center.flatten(), 10, (0, 255, 0), -1)
    
    if(Thymio.keypoints is not None):
        for i in range(len(Thymio.keypoints)):
            cv2.circle(image_cnt, Thymio.keypoints[i], 10, (200, 240, 190), -1)
        cv2.circle(image_cnt, Thymio.target_keypoint, 10, (0, 255, 255), -1)

    radius = 1.5 * camera.size_aruco
    if Thymio.Thymio_detected:
        Thymio_nose = radius * np.array(
            [np.cos(Thymio.xytheta_meas[2]), np.sin(Thymio.xytheta_meas[2])]
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

    # Kalman:
    # 2sigma-confidence Position (95%)
    
    eigenvalues, eigenvectors = np.linalg.eigh(Thymio.kalman_P[:2,:2])# Compute max variance direction and value

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    max_variance_direction = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])) 

    # Draw the variance ellipse
    cv2.ellipse(image_cnt,
                Thymio.xytheta_est[:2].astype(int),
                (2 * np.sqrt(eigenvalues[0]).astype(int), 2 * np.sqrt(eigenvalues[1]).astype(int)),
                max_variance_direction, 0, 360, (0, 255, 255), 2)
    '''
    cv2.circle(
        image_cnt,
        Thymio.xytheta_est[:2].astype(int),
        (2 * np.sqrt(Thymio.kalman_P[1, 1]) * Thymio.pixbymm).astype(int),
        (0, 255, 255),
        2,
    )
    '''
    # Angle:
    # sigma-confidence arc (95%)
    # Start of the arc:
    start_angle = np.max([np.degrees(Thymio.xytheta_est[2]) - np.degrees(
        2 * np.sqrt(Thymio.kalman_P[2, 2])
    ),0])
    # End of the arc
    end_angle = np.min([np.degrees(Thymio.xytheta_est[2]) + np.degrees(
        2 * np.sqrt(Thymio.kalman_P[2, 2])
    ),360])

    cv2.ellipse(
        image_cnt,
        Thymio.xytheta_est[:2].astype(int),
        (radius.astype(int), radius.astype(int)),
        0,
        start_angle,
        end_angle,
        (255, 0, 127),
        2,
    )
    cv2.imshow("Camera View", image_cnt)
    cv2.waitKey(1)


def draw_history(camera, Thymio, path_img, keypoints):
    image_cnt = camera.persp_image.copy()
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
    for i in range(len(keypoints)):
        cv2.circle(image_cnt, keypoints[i], 5, (200, 240, 190), -1)
    cv2.circle(image_cnt, camera.goal_center.flatten(), 10, (0, 255, 0), -1)

    # Draw history
    cv2.polylines(
        image_cnt,
        [Thymio.xytheta_meas_hist[:, :2].astype(int).reshape(-1, 1, 2)],
        isClosed=False,
        color=(255, 0, 255),
        thickness=2,
    )
    cv2.polylines(
        image_cnt,
        [Thymio.xytheta_est_hist[:, :2].astype(int).reshape(-1, 1, 2)],
        isClosed=False,
        color=(0, 255, 255),
        thickness=2,
    )
    cv2.imshow("History", image_cnt)
    cv2.waitKey(1)



def find_color_thresholds(bgr_image, bl_init, gl_init, rl_init, bh_init, gh_init, rh_init, filename):
    height, width, _ = bgr_image.shape

    B_LOW_slider = IntSlider(value=bl_init, min=0, max=255, step=1, description='Blue Low')
    G_LOW_slider = IntSlider(value=gl_init, min=0, max=255, step=1, description='Green Low')
    R_LOW_slider = IntSlider(value=rl_init, min=0, max=255, step=1, description='Red Low')
    B_HIGH_slider = IntSlider(value=bh_init, min=0, max=255, step=1, description='Blue High')
    G_HIGH_slider = IntSlider(value=gh_init, min=0, max=255, step=1, description='Green High')
    R_HIGH_slider = IntSlider(value=rh_init, min=0, max=255, step=1, description='Red High')
    x_slider = IntSlider(value=width // 2, min=0, max=width - 1, step=1, description='X Coordinate')
    save_button = ipywidgets.Button(description=f"Save Thresholds")
    output = Output()
    
    def update(_):
        with output:
            output.clear_output(wait=True)
            B_LOW = B_LOW_slider.value
            G_LOW = G_LOW_slider.value
            R_LOW = R_LOW_slider.value
            B_HIGH = B_HIGH_slider.value
            G_HIGH = G_HIGH_slider.value
            R_HIGH = R_HIGH_slider.value
            x = x_slider.value
            
            lower_t = (B_LOW, G_LOW, R_LOW)
            upper_t = (B_HIGH, G_HIGH, R_HIGH)
            mask = cv2.inRange(bgr_image, lower_t, upper_t)
            t_img = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
    
            b_values = bgr_image[:, x, 0]
            g_values = bgr_image[:, x, 1]
            r_values = bgr_image[:, x, 2]
    
            fig = plt.figure(figsize=(18, 12))
            fig.suptitle(f"Threshold Visualization (x={x})", fontsize=16)
    
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
            ax1.set_title("Input Image")
            ax1.axis("on")
            ax1.axvline(x=x, color='yellow') 
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
            ax2.set_title("Thresholded Image")
            ax2.axis("on")
            ax2.axvline(x=x, color='yellow') 
    
            gs_bottom = gs[1, :].subgridspec(1, 3)
            ax_r = fig.add_subplot(gs_bottom[0])
            ax_r.plot(range(height), r_values, color='red')
            ax_r.set_xlabel("y")
            ax_r.set_ylabel("Pixel intensity")
            ax_g = fig.add_subplot(gs_bottom[1])
            ax_g.plot(range(height), g_values, color='green')
            ax_g.set_xlabel("y")
            ax_b = fig.add_subplot(gs_bottom[2])
            ax_b.plot(range(height), b_values, color='blue')
            ax_b.set_xlabel("y")
    
            plt.tight_layout()
            plt.show()

    def save_thresholds(_):
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        if os.path.basename(script_dir) == "utils":
            utils_dir = script_dir
        else:
            utils_dir = os.path.join(script_dir, "utils")
        filepath = os.path.join(utils_dir, filename)
        B_LOW = B_LOW_slider.value
        G_LOW = G_LOW_slider.value
        R_LOW = R_LOW_slider.value
        B_HIGH = B_HIGH_slider.value
        G_HIGH = G_HIGH_slider.value
        R_HIGH = R_HIGH_slider.value
        lower_t = (B_LOW, G_LOW, R_LOW)
        upper_t = (B_HIGH, G_HIGH, R_HIGH)
        thresholds = {lower_t, upper_t}
        with open(filepath, 'w') as f:
            f.write(str(thresholds))

    B_LOW_slider.observe(update, 'value')
    G_LOW_slider.observe(update, 'value')
    R_LOW_slider.observe(update, 'value')
    B_HIGH_slider.observe(update, 'value')
    G_HIGH_slider.observe(update, 'value')
    R_HIGH_slider.observe(update, 'value')
    x_slider.observe(update, 'value')
    save_button.on_click(save_thresholds)

    sliders = VBox([B_LOW_slider, G_LOW_slider, R_LOW_slider, B_HIGH_slider, G_HIGH_slider, R_HIGH_slider, x_slider, save_button])
    display(sliders, output)
    update(None)