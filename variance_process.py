import numpy as np
import tdmclient as t
import logging
import time

R_WHEEL = 43  # wheels radius [mm]
L_AXIS = 92  # wheel axis length [mm]
SPEED_LIMIT = 500  # PWM
SPEED_SCALING_FACTOR = 500 / (200 / R_WHEEL)  # Scaling factor for speed
SPEED_SCALING_FACTOR_Kalman = 500 / 200
delta_t =10  # Time step in seconds


def normalize_angle(angle):  # restricts angle [rad] between -pi and pi
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def main():
    logging.basicConfig(level=logging.INFO)
    client = t.ClientAsync()

    logging.info("Looking for a Thymio robot...")
    node = t.aw(client.wait_for_node())
    if not node:
        logging.error("Thymio not connected.")
        return

    logging.info("Thymio connected.")
    t.aw(node.lock())
    t.aw(node.wait_for_variables(["motor.left.speed", "motor.right.speed"]))
    v_Left = 50
    v_Right = 50

    # Kalman filter initialization
    kalman_wheel_base = 92  # mm
    xytheta_est = np.array([0, 0, 0]) 
    v_L = v_Left / SPEED_SCALING_FACTOR_Kalman  # go from pwm to mm/s
    v_R = v_Right / SPEED_SCALING_FACTOR_Kalman

    theta = xytheta_est[2]

    # Compute linear and angular velocities
    v = (v_R + v_L) / 2
    omega = (v_L - v_R) / kalman_wheel_base

    # Update state
    delta_theta = omega *delta_t
    theta_mid = (
        theta + delta_theta / 2
    )  # midpoint method (the robot is turning so we take avg angle)
    

    t.aw(node.register_events([("speed", 2)]))
    program = """
    onevent speed
        motor.left.target = event.args[0]
        motor.right.target = event.args[1]
    """
    t.aw(node.compile(program))
    t.aw(node.run())
    logging.info("Program uploaded to Thymio.")

    # Control loop
    start_time = time.time()  # Record the start time
    elapsed_time = 0

    while elapsed_time < 10:  # Loop for 5 seconds
        # Fetch motor speeds
        state = {
            "motor.left.speed": node["motor.left.speed"],
            "motor.right.speed": node["motor.right.speed"],
        }
        v_L = state["motor.left.speed"]
        v_R = state["motor.right.speed"]

        # Log or print elapsed time
        elapsed_time = time.time() - start_time
        if elapsed_time> 9.95:
            logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        

        # Send commands to Thymio
        t.aw(node.send_events({"speed": [int(v_Left), int(v_Right)]}))

    # Stop the Thymio after 5 seconds
    logging.info("Stopping Thymio...")
    t.aw(node.send_events({"speed": [0, 0]}))
    delta_x = v * np.cos(theta_mid) * elapsed_time
    delta_y = v * np.sin(theta_mid) * elapsed_time
    xytheta_est = xytheta_est + np.array([delta_x, delta_y, delta_theta])

    # Normalize angle to [-pi, pi]
    xytheta_est[2] = normalize_angle(xytheta_est[2])

    print (xytheta_est)
    logging.info("Thymio has stopped.")


if __name__ == "__main__":
    main()
