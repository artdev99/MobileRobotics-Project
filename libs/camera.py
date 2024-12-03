import cv2
import numpy as np
import matplotlib.pyplot as plt 

CAMERA_INDEX = 1 # Aukey camera
RESOLUTION = (1920,1080)
FPS = 30
NOISE_THRESHOLD = 1000

def get_camera_url(login=None, password=None, ip=None):
    if ip is None:
        return None
    if login is None or password is None:
        return f"https://{ip}:8080/video"
    return f"https://{login}:{password}@{ip}:8080/video"

class CameraProcessor:
    def __init__(self, camera_index=CAMERA_INDEX, resolution=RESOLUTION, fps=FPS, camera_url=None, noise_threshold=NOISE_THRESHOLD):
        
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.camera_url = camera_url
        self.noise_threshold = noise_threshold

        if self.camera_url:
            camera = cv2.VideoCapture(self.camera_url)
            self.setup_camera(None, None)
        else:
            camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.setup_camera(self.resolution, self.fps)

        if not camera.isOpened():
            camera.release()
            raise IOError("Camera could not be opened")
        
        # Warmup: camera needs to do lighting and white balance adjustments
        for _ in range(50):
            ret, _ = camera.read()
            if not ret:
                camera.release()
                raise IOError("Failed to capture frame during warmup")

        self.capture_image()

    def capture_image(self):
        if self.camera_url:
            camera = cv2.VideoCapture(self.camera_url)
        else:
            camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        ret, image = camera.read()        
        if not ret:
            camera.release()
            raise IOError("Failed to get image")
        return image

    def setup_camera(self, resolution=None, fps=None):
        """
        1920, 1080 (16:9)
        1280, 720  (16:9)
        640, 380   (16:9)
        320, 240   (4:3)
        """
        if self.camera_url:
            camera = cv2.VideoCapture(self.camera_url)
        else:
            camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if resolution is not None:
            width, height = resolution
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        if fps is not None:
            camera.set(cv2.CAP_PROP_FPS, fps)

        w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = int(camera.get(cv2.CAP_PROP_FPS))

        print(f"Resolution is: {w}x{h}")
        print(f"FPS is: {f}") 
        
    def live_feed(self, resize_factor=0.5):
        if self.camera_url:
            camera = cv2.VideoCapture(self.camera_url)
        else:
            camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        print("Press 's' to save an image or 'q' to quit.")
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to get image")
                break
            temp = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            cv2.imshow("Live Feed", temp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                image = frame.copy()
                print("Image saved to variable 'image'.")
            elif key == ord('q'):
                print("Exiting.")
                cv2.destroyWindow("Live Feed")
                break
        return image

    def camera_obstructed(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        var = np.var(gray)
        #print(var, self.noise_threshold)
        return var < self.noise_threshold
    
    def show_image(self, image, fig_size=(6,6), color="BGR", _axis=True, _title=None, _cmap=None):
        plt.figure(figsize=fig_size)
        if color == "BGR":
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=_cmap)
        else:
            if color != "RGB":
                print(f"color {color} not recognized, applying default RGB")
            plt.imshow(image, cmap=_cmap)
        if _title is not None:
            plt.title(_title)
        if not _axis:
            plt.axis('off')
        plt.show()

