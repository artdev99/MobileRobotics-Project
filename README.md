# DEADLINE: 5/12 23:00

Group 47
| Name      | ID      |
|-----------|---------|
| Martin    | 340936  |
| Laetitia  | XXXXXX  |
| Arthur    | 300443  |
| Amandine  | XXXXXX  |

## TODO
Code in Pythin using async programming (no `%%run_python` or `%%run_aseba`)
### High Priority
- Mapping à partir de la caméra + Localisation du robot (Martin et Arthur)
  - Picture the scene
  - Get a 2D scene (adjust the angle)
  - Identify (using Leds or color of env/robot/curve of thymio)
- Astolfi (contrôle) (Amandine)
- Kalman (camera + motor speed)

### Later
- Live Visualization
- A* / gridsearch
- Local Avoidance
- Hide Camera

### To lose time
- impossible escape (local avoidance)
- kidnapping (global localization)
- add a second thymio

## Deliverables
A **Jupyter notebook** which serves as a report. This must contain the information regarding :
- The **members of the group**, it’s helpful to know who you are when we go over the report.
- An **introduction to your environment** and to the **choices you made**.
- Sections that go into a bit **more detail regarding the implementation** and are accompanied by the code required to **execute the different modules independently**. What is important is not to simply describe the code, which should be readable, but describe what is not in the code: the theory behind, the choices made, the measurements, the choice of parameters etc. Do not forget to **cite your sources** ! You can of course show how certain modules work in **simulation here, or with pre-recorded data**.
- A section which is used to **run the overall project** and where we can see the path chosen, where the system believes the robot is along the path before and after filtering etc… This can also be done **in a .py file** if you prefer. Just make sure to reference it in the report so we can easily have a look at it.
- Conclusion: The code used to execute the overall project. Do not hesitate to **make use of .py files** that you import in the notebook. The whole body of code does not have to be in the Jupyter notebook, on the contrary! Make sure the **code clean and well documented**. Somebody who starts to browse through it should easily understand what you are doing.
