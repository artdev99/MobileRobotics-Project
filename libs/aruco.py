import cv2
import numpy as np

# ArUco
CORNERS_ARUCO_ID = [0, 1, 2, 10] # top-left, bottom-left, bottom-right, top-right
ARUCO_CORNERS_ORDER = [2, 1, 0, 3]
ARUCO_SIZE_MM = 70 # mm

class ArucoDetector:
    def __init__(self, corners_aruco_id = CORNERS_ARUCO_ID, corners_order = ARUCO_CORNERS_ORDER, aruco_size_mm = ARUCO_SIZE_MM):
        self.corners_aruco_id = corners_aruco_id
        self.corners_order = corners_order
        self.aruco_size_mm = aruco_size_mm
        self.parameters = cv2.aruco.DetectorParameters()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_size_px = None
        self.max_width_perspective = None
        self.max_height_perspective = None
        self.M = None
        self.pixbymm = None
        
    def order_points(self, corners):
        rect = np.zeros((4, 2), dtype="float32")
        _sum = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        rect[0] = corners[np.argmin(_sum)]
        rect[2] = corners[np.argmax(_sum)]
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]
        return rect
        
    def compute_destination_size(self, ordered_corners):
        (top_left, top_right, bottom_right, bottom_left) = ordered_corners
        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        max_width = max(int(width_top), int(width_bottom))
        height_left = np.linalg.norm(top_left - bottom_left)
        height_right = np.linalg.norm(top_right - bottom_right)
        max_height = max(int(height_left), int(height_right))
        return max_width, max_height

    def find_aruco_corners_size(self, image):
        gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cv2.__version__ == "4.10.0":
            _aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            corners, ids, _ = _aruco_detector.detectMarkers(gray_img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray_img, self.aruco_dict, parameters=self.parameters)
        if ids is None or len(ids) < len(self.corners_aruco_id):
            raise ValueError("Not enough ArUco markers detected")
        inner_corners = []

        for marker_id, corner_pos in zip(self.corners_aruco_id, self.corners_order):
            idx_array = np.where(ids == marker_id)[0]
            if idx_array.size == 0:
                raise ValueError(f"Marker {marker_id} not found")
            idx = idx_array[0]
            inner_corners.append(corners[idx][0,corner_pos,:])
        size_aruco_px=[]
        for i in range(4):
            side_lengths = [
                np.linalg.norm(corners[i][0,0,:] - corners[i][0,1,:]),  # Top side
                np.linalg.norm(corners[i][0,1,:] - corners[i][0,2,:]),  # Right side
                np.linalg.norm(corners[i][0,2,:] - corners[i][0,3,:]),  # Bottom side
                np.linalg.norm(corners[i][0,3,:] - corners[i][0,0,:])   # Left side
            ]
            size_aruco_px.append(np.mean(side_lengths))
        return np.array(inner_corners), np.mean(size_aruco_px)

    def correct_perspective(self, image, find_matrix=False):
        if find_matrix:
            corners, self.aruco_size_px = self.find_aruco_corners_size(image)
            ordered_corners = self.order_points(corners)
            self.max_width_perspective, self.max_height_perspective = self.compute_destination_size(ordered_corners)
            destination_corners = np.array([
                [0, 0],
                [self.max_width_perspective - 1, 0],
                [self.max_width_perspective - 1, self.max_height_perspective - 1],
                [0, self.max_height_perspective - 1]], dtype="float32")
            self.M = cv2.getPerspectiveTransform(ordered_corners, destination_corners)
            self.pixbymm = self.aruco_size_px / self.aruco_size_mm

        return cv2.warpPerspective(image, self.M, (self.max_width_perspective, self.max_height_perspective), flags=cv2.INTER_LINEAR)



