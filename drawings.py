import numpy as np
import cv2

def draw_on_image(camera, Thymio, kalman, path_img):
    image_cnt = camera.persp_image.copy()
    cv2.drawContours(image_cnt, camera.goal_cnt, -1, (0,255,0), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt, -1, (0,0,255), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt_expnded, -1, (0,100,255), 3)
    cv2.polylines(image_cnt, [path_img.T.reshape(-1,1,2)], isClosed=False, color=(255, 0, 0), thickness=3)
    cv2.circle(image_cnt,camera.goal_center.flatten(), 10, (0,255,0), -1)
    for i in range(len(Thymio.keypoints)):
        cv2.circle(image_cnt, Thymio.keypoints[i], 10, (200, 240, 190), -1)
    cv2.circle(image_cnt, Thymio.target_keypoint, 10, (0, 255, 255), -1)

    radius = 1.5 * camera.size_aruco
    if Thymio.Thymio_detected:
        Thymio_nose = radius*np.array([np.cos(Thymio.xytheta_meas[2]), np.sin(Thymio.xytheta_meas[2])]) #thymio is approx 1.5 aruco size
        Thymio_nose = Thymio_nose + Thymio.xytheta_meas[:2]
        cv2.arrowedLine(image_cnt, Thymio.xytheta_meas[:2].astype(int), Thymio_nose.astype(int), (255, 0, 255), 2, tipLength=0.2)
    
    #Kalman:
    #2sigma-confidence Position (95%)

    cv2.circle(image_cnt, Thymio.xytheta_est[:2].astype(int), (2*np.sqrt(kalman.P[2,2])*Thymio.pixbymm).astype(int), (0, 255, 255), 2)
    #Angle:
    # sigma-confidence arc (95%)
    start_angle = np.degrees(Thymio.xytheta_est[2]) - np.degrees(2*np.sqrt(kalman.P[2,2]))  # Start of the arc
    end_angle = np.degrees(Thymio.xytheta_est[2]) + np.degrees(2*np.sqrt(kalman.P[2,2]))    # End of the arc

    cv2.ellipse(image_cnt, Thymio.xytheta_est[:2].astype(int), (radius.astype(int), radius.astype(int)), 0, start_angle, end_angle, (255, 0, 127), 2)
    
    cv2.imshow('Camera View', image_cnt)
    cv2.waitKey(1)

def draw_history(camera,Thymio,path_img, keypoints):
    image_cnt = camera.persp_image.copy()
    cv2.drawContours(image_cnt, camera.goal_cnt, -1, (0,255,0), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt, -1, (0,0,255), 3)
    cv2.drawContours(image_cnt, camera.obstacle_cnt_expnded, -1, (0,100,255), 3)
    cv2.polylines(image_cnt, [path_img.T.reshape(-1,1,2)], isClosed=False, color=(255, 0, 0), thickness=3)
    for i in range(len(keypoints)):
        cv2.circle(image_cnt, keypoints[i], 5, (200, 240, 190), -1)
    cv2.circle(image_cnt,camera.goal_center.flatten(), 10, (0,255,0), -1)

    #Draw history
    cv2.polylines(image_cnt, [Thymio.xytheta_meas_hist[:,:2].astype(int).reshape(-1,1,2)], isClosed=False, color=(255, 0, 255), thickness=2)
    cv2.polylines(image_cnt, [Thymio.xytheta_est_hist[:,:2].astype(int).reshape(-1,1,2)], isClosed=False, color=(0, 255, 255), thickness=2)
    cv2.imshow('History', image_cnt)
    cv2.waitKey(1)
