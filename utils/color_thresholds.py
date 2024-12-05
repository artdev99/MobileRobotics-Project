import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
from ipywidgets import IntSlider, VBox, Output
from IPython.display import display
import os

def get_camera_url(login=None, password=None, ip=None):
    if ip is None:
        return None
    if login is None or password is None:
        return f"https://{ip}:8080/video"
    return f"https://{login}:{password}@{ip}:8080/video"

def get_camera(camera_url=None, camera_index=1):
    if camera_url: 
        camera = cv2.VideoCapture(camera_url)
    else:
        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise IOError("Camera could not be opened")
    return camera

def load_thresholds(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "utils":
        filepath = os.path.join(script_dir, filename)
    else:
        filepath = os.path.join(script_dir, "utils", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Threshold file not found: {filepath}")
    with open(filepath, 'r') as f:
        content = f.read().strip()
        content = content[1:-1] # remove {}
        lower_t, upper_t = content.split("), (", 3) 
        lower_t = lower_t[1:].split(", ")
        lower_t = tuple(map(int, lower_t))
        upper_t = upper_t[:-1].split(", ")
        upper_t = tuple(map(int, upper_t))
        return np.array(lower_t + upper_t) # (6,)
    

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