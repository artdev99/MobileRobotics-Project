import time
from time import sleep
import numpy as np

start_time = time.time()
position_x = []
position_y = []
position_theta = []
while time.time() - start_time < 5:
    image, Thymio_xytheta, Thymio_detected, Thymio_size, Thymio_nose, Thymio_cnt = (
        update_vision(
            cam,
            sigma,
            epsilon,
            mat_persp,
            max_width_persp,
            max_height_persp,
            thresh_Thymio,
            Thymio_size,
        )
    )
    position_x.append((Thymio_xytheta.flatten())[0])
    position_y.append((Thymio_xytheta.flatten())[1])
    position_theta.append((Thymio_xytheta.flatten())[2])
    sleep(0.15)

vx = np.var(position_x)
vy = np.var(position_y)
vt = np.var(position_theta)

print("vx:", vx)
print("vy:", vy)
print("vt:", vt)
