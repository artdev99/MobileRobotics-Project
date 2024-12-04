import numpy as np
import cv2
from tdmclient import ClientAsync, aw
from camera_class import *
from thymio_class import *
from path import *
from braitenberg import *
from kalman import *
from motors import *
from buttons import*

###########################################################
#Parameters
###########################################################
CAMERA_INDEX = 1 #0 if no webcam
CORNER_ARUCO_ID = [0, 1, 2, 10] #top-left, bottom-left, bottom-right, top-right
CORNER_ARUCO_SIZE = 65          #[mm]
MIN_SIZE = 500 #minimum blob size
COLOR_OBSTACLE = np.array([[30,20,120,65,50,170]]) #BGR
COLOR_GOAL = np.array([30,40,20,80,150,65])        #BGR
THYMIO_ID = 9
GRID_L = 400 #[pixels]
GRID_W = 300 #[pixels]
DISTANCE_THRESH = 65 #[mm]
###########################################################
#Main Code
###########################################################
from tdmclient import ClientAsync, aw
client = ClientAsync()

async def main():
    node = await client.wait_for_node()
    aw(node.lock())
    
    print("Press forward button to start the program")
    beginning = False
    while(beginning == False) :
        beginning = await wait_for_start_button(node, client)
        time.sleep(0.3)
    cv2.destroyAllWindows()
    print("Starting the program")

    #Camera initialization
    cam = Camera_class(CAMERA_INDEX,CORNER_ARUCO_ID,CORNER_ARUCO_SIZE, MIN_SIZE, COLOR_OBSTACLE, COLOR_GOAL)

    #Thymio initialization
    Thymio = Thymio_class(THYMIO_ID,cam)

    #Kalman initialization
    kalman = Kalman_class(cam)

    path_planning = True #We want to have the path
    local_avoidance = False
    step = 0
    
    while True :    
        step = step + 1
        #Update Image
        cam.get_image()
        cam.correct_perspective_aruco(get_matrix = False)
        #Path Planning
        if path_planning:
            if Thymio.target_keypoint is None or not np.any(Thymio.target_keypoint):
                do_plot = True
            grid = discretize_image_eff(cam.thresholded_image,GRID_L, GRID_W)
            #Careful! Image frame's first coord (x) is pointing right but in a matrix the first coordinate (rows) is pointing down so they must be inverted
            found, path, _, _ = a_star_search(grid, grid1_coord2grid2_coord(np.array([Thymio.xytheta_est[1], Thymio.xytheta_est[0]]), cam.persp_image, grid), grid1_coord2grid2_coord(np.array([cam.goal_center[1], cam.goal_center[0]]), cam.persp_image,grid), do_plot)
            
            if(not found):
                print("couldn't find path, stopping the mission")
                aw(node.stop())
                aw(node.unlock())
                #draw_history(cam,Thymio,path_img, keypoints) #crash if no path ever
                break
            
            #Convert path coordinates for plotting
            path_img = grid1_coord2grid2_coord(path, grid, cam.persp_image)
            path_img = path_img[::-1]
            
            keypoints = find_keypoints(path_img)
            Thymio.target_keypoint = keypoints[0]
            Thymio.keypoints=keypoints[1:]
            
            do_plot = False
            path_planning = False
 
        Thymio.Thymio_position_aruco(cam.persp_image)
        kalman.delta_time_update()      

        #Kalman Filter
        v_L, v_R = await kalman.gather_data(node)
        kalman.predict_state(Thymio.xytheta_est,v_L,v_R)
        if Thymio.Thymio_detected: #only update if Thymio detected
            kalman.update_state(Thymio.xytheta_est, Thymio.xytheta_meas)

        #Update history for final plot
        if((step % 3) == 0):
            Thymio.xytheta_meas_hist = np.vstack((Thymio.xytheta_meas_hist, Thymio.xytheta_meas))
            Thymio.xytheta_est_hist = np.vstack((Thymio.xytheta_est_hist, Thymio.xytheta_est))
        
        #Obstacle detection
        prox_values = await get_prox(node, client)
        if (check_obstacle(prox_values)):
            print("obstacle")
            local_avoidance = True
            while (check_obstacle(prox_values)):
                prox_values = await get_prox(node, client)
                await set_motors(node, avoid_obstacle(prox_values))
            await set_motors(node, 50, 50)
            #time.sleep(0.2)
            draw_on_image(cam, Thymio, path_img)
            continue
        else:
            if local_avoidance:
                print("recalculating path")
                path_planning = True
                local_avoidance = False
                draw_on_image(cam, Thymio, path_img)
                continue
            else:
                if((step % 5) == 0):
                    #print("distance to keypoint: ", distance_to_goal(cam.pixbymm))
                    if((Thymio.distance_to_goal()) < DISTANCE_THRESH):
                        if(len(Thymio.keypoints) <= 1): #Thymio found the goal
                            print("Mission accomplished") 
                            aw(node.stop())
                            aw(node.unlock())
                            draw_history(cam, Thymio, path_img, keypoints)
                            break
                        Thymio.keypoints = Thymio.keypoints[1:]
                        Thymio.target_keypoint = Thymio.keypoints[0]
                    await set_motors(node, Thymio.motion_control())
                
                draw_on_image(cam, Thymio, path_img)
        if(await check_stop_button(node, client)):
            aw(node.stop())
            aw(node.unlock())
            draw_history(cam, Thymio,path_img, keypoints)
            break
    cam.cam.release()
    #cv2.destroyAllWindows()

# Run the main asynchronous function
while True:
    client.run_async_program(main)
    