import numpy as np
import cv2

def draw_on_image(camera, Thymio, path_img,capture=False):
    image_cnt = camera.persp_image.copy()
    cv2.drawContours(image_cnt, camera.goal_cnt, -1, (0, 255, 0), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt, -1, (122, 43, 46), 3) 
    cv2.drawContours(image_cnt, camera.obstacle_cnt_expnded, -1, (196, 176, 68), 3)
    
    if path_img is not None: 
        cv2.polylines(
            image_cnt,
            [path_img.T.reshape(-1, 1, 2)],
            isClosed=False,
            color=(0, 0, 255), 
            thickness=3,
        )
    cv2.circle(image_cnt, camera.goal_center.flatten(), 10, (0, 255, 0), -1)
    
    #Thymio measured
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
            (0, 128, 255),
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
                max_variance_direction, 0, 360, (255,0, 127), 2)

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
        (255, 0, 255),
        2,
    )
    
    if(Thymio.keypoints is not None):
        for i in range(len(Thymio.keypoints)):
            cv2.circle(image_cnt, Thymio.keypoints[i], 10, (200, 240, 190), -1)
        cv2.circle(image_cnt, Thymio.target_keypoint, 10, (0, 255, 255), -1)
    
    cv2.imshow("Camera View", image_cnt)
    cv2.waitKey(1)
    if capture:
        cv2.imwrite("Camera View.png", image_cnt)



def draw_history(camera, Thymio, path_img, keypoints):
    image_cnt = camera.persp_image.copy()
    cv2.drawContours(image_cnt, camera.goal_cnt, -1, (0, 255, 0), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt, -1, (122, 43, 46), 3) 
    cv2.drawContours(image_cnt, camera.obstacle_cnt_expnded, -1, (196, 176, 68), 3)
    for i in range(len(path_img)):
        cv2.polylines(
            image_cnt,
            [path_img[i].T.reshape(-1, 1, 2)],
            isClosed=False,
            color=(0, 20*i, 255),
            thickness=3,
        )

    # Draw history
    for i in range(Thymio.xytheta_est_hist.shape[0]):
        cv2.circle(image_cnt, Thymio.xytheta_est_hist[i, :2].astype(int), 3, (255, 0, 127), -1)
    for i in range(Thymio.xytheta_meas_hist.shape[0]):
        cv2.circle(image_cnt, Thymio.xytheta_meas_hist[i, :2].astype(int), 2, (255, 0, 255), -1)
        
    for i in range(len(keypoints)):
        cv2.circle(image_cnt, keypoints[i], 5, (200, 240, 190), -1)
    cv2.circle(image_cnt, camera.goal_center.flatten(), 10, (0, 255, 0), -1)

    """
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
        color=(255, 0, 127),
        thickness=2,
    )"""
    cv2.imshow("History", image_cnt)
    cv2.waitKey(1)
    cv2.imwrite("History.png","History", image_cnt)