import numpy as np
import cv2
from tdmclient import ClientAsync
from final_camera_class import *
from final_Thymio_class import *
from final_path import *
from motion import *

###########################################################
#Parameters
###########################################################
camera_index=1 #0 if no webcam
corner_aruco_id=[0, 1, 2, 10] #top-left, bottom-left, bottom-right, top-right
corner_aruco_size=70 #mm
min_size=5000 #minimum blob size
thresh_obstacle=np.array([0,0,120,0,0,140]) #BGR
thresh_goal=np.array([0,120,0,0,140,0]) #BGR
Thymio_id=9
grid_size0=450 #blocks? TBD numbers of blocks or pixels?
grid_size1=300
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
            cam=camera_class(camera_index,corner_aruco_id,corner_aruco_size,min_size, thresh_obstacle, thresh_goal)

            #Thymio initialization
            Thymio=Thymio_class(Thymio_id,cam.persp_image)
            

            Path_planning=True #We want to have the path
            step = 0
            while True:
                step = step + 1
                #Update Image
                cam.get_image()
                cam.correct_perspective_aruco(get_matrix=False)
                #Path Planning
                if Path_planning:
                    if Thymio.target_keypoint==None: #only possible at first iteration to not take time later
                        do_plot=True
                    grid=discretize_image_eff(cam.thresholded_image,grid_size)
                    #Careful! Image frame's first coord (x) is pointing right but in a matrix the first coordinate (rows) is pointing down so they must be inverted
                    path= a_star_search(grid, grid1_coord2grid2_coord(np.array([Thymio.xytheta_est[1],Thymio.xytheta_est[0]]),cam.persp_image,grid), grid1_coord2grid2_coord(np.array([cam.goal_center[1],cam.goal_center[0]]),cam.persp_image,grid),do_plot)
                    
                    # Convert path coordinates for plotting
                    path_img = grid1_coord2grid2_coord(path, grid, cam.perspimage)
                    path_img = path_img[::-1]

                    keypoints=find_keypoints(path_img,ANGLE_THRESHOLD,STEP,COUNTER_THRESHOLD)
                    keypoints=keypoints[np.linalg.norm(keypoints-Thymio.xytheta_est[:2],axis=1)<keypoint_dist_thresh,:] #Keep only far keypoints
                    Thymio.target_keypoint=Thymio.keypoints[0,:]
                    Thymio.keypoints=keypoints[1:,:]
                    do_plot=False
                    Path_planning=False
                
                #Thymio Position and motor 
                Thymio.Thymio_position_aruco(cam.persp_image)
                Thymio.delta_time_update()
                #TBD await get motor speed something
                if((step % 5)==0) :
                    x_cm, y_cm, theta_rad, x_goal_cm, y_goal_cm = adjust_units(Thymio.xytheta_meas, Thymio.target_keypoint, cam.pixbymm)
                    v_m = motion_control(x_cm, y_cm, theta_rad, x_goal_cm, y_goal_cm)
                    print("x_cm = ", x_cm, " y_cm = ", y_cm)
                    print ("v_m = ", v_m)
                    await node.set_variables(v_m)

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
                    draw_on_image(cam,Thymio,path_img)
                    continue
                else:
                    if Thymio.local_avoidance:
                        Path_planning=True
                        draw_on_image(cam,Thymio,path_img)
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
                        draw_on_image(cam,Thymio,path_img)
            cam.cam.release()
            cv2.destroyAllWindows()

    # Run the main asynchronous function
    client.run_async_program(main)