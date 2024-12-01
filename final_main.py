import numpy as np
import cv2
from tdmclient import ClientAsync
import time
from final_classes import *
from final_functions import *


###########################################################
#Parameters
###########################################################
camera_index=1 #0 if no webcam
corner_aruco_id=[0, 1, 2, 10] #top-left, bottom-left, bottom-right, top-right
min_size=5000 #minimum blob size
thresh_obstacle=np.array([0,0,120,0,0,140]) #BGR
thresh_goal=np.array([0,120,0,0,140,0]) #BGR
Thymio_id=9
grid_size=300 #pix
ANGLE_THRESHOLD = np.radians(40)   #threshold under which changes of directions are ignored [rad]
STEP = 3                           #step (in number of cells) between each cell we study
COUNTER_THRESHOLD = 3              #max number of steps between keypoints
keypoint_dist_thresh=100 #pix TBD

###########################################################
#Main Code
###########################################################
# Start the asynchronous client
with ClientAsync() as client:
    async def main():
        async with client.lock() as node:
            #Camera initialization
            cam=camera_class(camera_index,corner_aruco_id,min_size, thresh_obstacle, thresh_goal)

            #Thymio initialization
            Thymio=Thymio_class(Thymio_id,cam.persp_image)
            

            Path_planning=True #We want to have the path

            while True:
            
                #Path Planning
                if Path_planning:
                    if Thymio.target_keypoint==None: #only possible at first iteration to not take time later
                        do_plot=True
                    grid=discretize_image_eff(cam.thresholded_image,grid_size)
                    #Careful! Image frame's first coord (x) is pointing right but in a matrix the first coordinate (rows) is pointing down so they must be inverted
                    path, _, _ = a_star_search(grid, grid1_coord2grid2_coord(np.array([Thymio.xytheta_est[1],Thymio.xytheta_est[0]]),cam.persp_image,grid), grid1_coord2grid2_coord(np.array([cam.goal_center[1],cam.goal_center[0]]),cam.persp_image,grid),do_plot)
                    keypoints=find_keypoints(path,ANGLE_THRESHOLD,STEP,COUNTER_THRESHOLD)
                    keypoints=keypoints[np.linalg.norm(keypoints-Thymio.xytheta_est[:2],axis=1)<keypoint_dist_thresh,:] #Keep only far keypoints
                    Thymio.target_keypoint=Thymio.keypoints[0,:]
                    Thymio.keypoints=keypoints[1:,:]
                    Path_planning=False
                
                #Thymio Position and motor 
                Thymio.Thymio_position_aruco(cam.persp_image)
                Thymio.delta_time_update()
                #TBD await get motor speed something
                if Thymio.Thymio_detected:
                    #TBD Thymio.kalmanfilter_detected don't forget variance of kalman filter in plot!!
                    pass
                else:
                    #TBD Thymio.kalmanfilter_not_detected
                    pass
                
                #Obstacle detection
                #TBD await get oprox sensor data
                obstacle=False #TBD
                if obstacle:
                    Thymio.local_avoidance=True
                    #TBD Thymio.local avoidance to update target motor speed
                    #TBD await set speed (thymio.speed)
                    #TBD call update plot function
                    continue
                else:
                    if Thymio.local_avoidance:
                        Path_planning=True
                        #TBD update plots
                        continue
                #Target Achieved?
                    else:
                        if np.linalg.norm(Thymio.xytheta_est[:2]-Thymio.target_keypoint)<keypoint_dist_thresh:
                            if Thymio.keypoints.size==0:
                                print("Goal Achieved!")
                                break
                            else: #Update target
                                Thymio.target_keypoint=Thymio.keypoints[0,:]
                                Thymio.keypoints=Thymio.keypoints[1:,:]
                        #Controller:
                        #TBD Thymio.astolfi get speed and set them
                        #TBD Update plots
            cam.cam.release()
            cv2.destroyAllWindows()

    # Run the main asynchronous function
    client.run_async_program(main)