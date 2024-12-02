import math
import numpy as np
from filterpy.kalman import KalmanFilter
import random

# Kalman Filter initial  
KFilter = KalmanFilter(dim_x=3, dim_z=3)

KFilter.F = np.array([[1, 0, 0],  # State transition matrix
                      [0, 1, 0],
                      [0, 0, 1]])

KFilter.H = np.array([[1, 0, 0],  # Measurement function
                      [0, 1, 0],
                      [0, 0, 1]])

KFilter.P = np.eye(3) * 9999  # Initial covariance matrix (uncertainty)
KFilter.R = np.diag([0.1, 0.1, 0.01])  # Measurement noise
KFilter.Q = np.diag([0.01, 0.01, 0.001])  # Process noise

delta_t = 0.15  # Time step
scale_motor_rad_sec = 40
radius = 0.02  # Wheel radius
distance_wheels = 0.095  # Distance between wheels
scale_camera = 1920 / 1.0  # Camera scale factor


def run_Kalman_filter(right_wheel_speed, left_wheel_speed, previous_angle, vision_measurements):
    global KFilter

    # Convert wheel speeds to radians per second
    right_wheel_speed = right_wheel_speed / scale_motor_rad_sec
    left_wheel_speed = left_wheel_speed / scale_motor_rad_sec

    Wheels_speed = np.array([[right_wheel_speed],
                             [left_wheel_speed]])
    Transition = np.array([
        [np.cos(previous_angle) * (delta_t / 2), np.cos(previous_angle) * (delta_t / 2)],
        [np.sin(previous_angle) * (delta_t / 2), np.sin(previous_angle) * (delta_t / 2)],
        [(-delta_t / distance_wheels), (delta_t / distance_wheels)]
    ]) * radius

    KFilter.predict(u=Wheels_speed, B=Transition)

    if vision_measurements is not None:
        measurement = np.array([vision_measurements[0] / scale_camera, vision_measurements[1] / scale_camera, vision_measurements[2]])
        KFilter.update(measurement)

    return np.array([KFilter.x[0] * scale_camera, KFilter.x[1] * scale_camera, KFilter.x[2]])


# Single Step Test (random values chosen)
if __name__ == "__main__":
    omega_right = 50  # Right wheel motor speed
    omega_left = 45   # Left wheel motor speed
    previous_angle = 0.1  # Previous orientation in radians
    vision_measurements = [1.0, 2.0, 0.5]  # Vision measurements (x, y, theta)

    noisy_omega_right = omega_right + random.gauss(0, 2)  # Add Gaussian noise
    noisy_omega_left = omega_left + random.gauss(0, 2)
    noisy_vision_x = vision_measurements[0] + random.gauss(0, 0.1)
    noisy_vision_y = vision_measurements[1] + random.gauss(0, 0.1)
    noisy_vision_angle = vision_measurements[2] + random.gauss(0, 0.01)

    estimate = run_Kalman_filter(
        right_wheel_speed=noisy_omega_right,
        left_wheel_speed=noisy_omega_left,
        previous_angle=previous_angle,
        vision_measurements=[noisy_vision_x, noisy_vision_y, noisy_vision_angle]
    )

    print(f"Estimated Position: {estimate}")
