# A3 landscape width  420mm
# A3 landscape height 297mm
# Aspect ratio 1.414141 (420/297) == (width/height)
a3_width = 420  # mm
a3_height = 297  # mm

# laptop screen
screen_width = 139.0  # mm
screen_height = screen_width / (420.0 / 297.0)

# IRL
robot_width = 110  # mm
robot_height = 110  # mm
robot_height_1 = 85  # mm
robot_height_2 = 25  # mm

# 420 = 139
# 110 = ?

screen_robot_width = int(robot_width * screen_width / a3_width)
screen_robot_height_1 = int(robot_height_1 * screen_height / a3_height)
screen_robot_height_2 = int(robot_height_2 * screen_height / a3_height)

print(screen_robot_width, screen_robot_height_1, screen_robot_height_2)
