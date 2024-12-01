
#MARKERS ARE 0 1 2 10




import numpy as np
import cv2

image=cv2.imread("aruco_test1.png")

print("Try to open camera")

cam = cv2.VideoCapture(1,cv2.CAP_DSHOW)

if not cam.isOpened(): 
    print("Camera could not be opened") 
    cam.release()
    exit()
print("Warming up the camera...")
for _ in range(50):  # Number of frames to discard (adjust if needed)
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture frame")
        break

print("Camera warmed up. Starting capture...")
# Initialize OpenCV window and matplotlib plot
ret, image = cam.read()
if not ret:
    print("Failed to capture image")
    cam.release()
    exit()
cv2.imshow("test",image)
cv2.waitKey(10)
# Initialize the detector parameters
parameters = cv2.aruco.DetectorParameters()
# Select the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters=parameters)

print(f"{len(corners)} corners found")


inner_corners = []

# Define the order of markers: top-left, bottom-left, bottom-right, top-right
marker_order = [0, 1, 2, 10]
aruco_corner = [2, 1, 0, 3]  # Bottom-right, top-right, top-left, bottom-left of each aruco

for marker_id, corner_pos in zip(marker_order, aruco_corner):
    idx = np.where(ids == marker_id)[0][0]
    # Get the inner corner
    inner_corners.append(corners[idx][0,corner_pos,:])
size_aruco=[]
for i in range(4):
    side_lengths = [
        np.linalg.norm(corners[i][0,0,:] - corners[i][0,1,:]),  # Top side
        np.linalg.norm(corners[i][0,1,:] - corners[i][0,2,:]),  # Right side
        np.linalg.norm(corners[i][0,2,:] - corners[i][0,3,:]),  # Bottom side
        np.linalg.norm(corners[i][0,3,:] - corners[i][0,0,:])   # Left side
    ]
    size_aruco.append(np.mean(side_lengths))

