# DEADLINE: 5/12 23:00 🎯

# MONDADA GOLDEN RULE: NO PLAGIARISM (or we all get fcked) 😐
They have a database of all the previous projects. They will compare our work to the databate, if there is a match, it's over 💀 <br>
If you use chatgpt, it's ok for prototyping, but for security **final code should be fully rewritten by us !!!**

# GOAL: REPRODUCE THE "FIND BEER" Project 🍺
- Using camera: create grid of env and detect:
  - robot (start position)
  - global obstacles
  - goal (end position)
- Plan to the goal using A*
- Kalman on motor speed + grid for precize localization of the robot
- Astolfi controler $(x,y,\theta)$ for the movement
- local avoidance (+ handle grid/map updates)

## TODO 🔨
Code in Pythin using async programming (no `%%run_python` or `%%run_aseba`)
### High Priority
- Mapping à partir de la caméra + Localisation du robot (Martin et Arthur)
  - Picture the scene
  - Get a 2D scene (adjust the angle)
  - Identify (using Leds or color of env/robot/curve of thymio)
- Astolfi (contrôle) (Amandine)
- Kalman (camera + motor speed)

### Later ⛏️
- Live Visualization
- A* / gridsearch
- Local Avoidance
- Hide Camera
- Start button

### To lose time 🌬️
- impossible escape (local avoidance)
- kidnapping (global localization)
- add a second thymio (dynamic local obstacle, dynamic goal $\rightarrow$ D*)

## Deliverables 📩
A **Jupyter notebook** which serves as a report. This must contain the information regarding :
- The **members of the group**, it’s helpful to know who you are when we go over the report.
- An **introduction to your environment** and to the **choices you made**.
- Sections that go into a bit **more detail regarding the implementation** and are accompanied by the code required to **execute the different modules independently**. What is important is not to simply describe the code, which should be readable, but describe what is not in the code: the theory behind, the choices made, the measurements, the choice of parameters etc. Do not forget to **cite your sources** ! You can of course show how certain modules work in **simulation here, or with pre-recorded data**.
- A section which is used to **run the overall project** and where we can see the path chosen, where the system believes the robot is along the path before and after filtering etc… This can also be done **in a .py file** if you prefer. Just make sure to reference it in the report so we can easily have a look at it.
- Conclusion: The code used to execute the overall project. Do not hesitate to **make use of .py files** that you import in the notebook. The whole body of code does not have to be in the Jupyter notebook, on the contrary! Make sure the **code clean and well documented**. Somebody who starts to browse through it should easily understand what you are doing.

Group 47🤺
| Name      | Sciper      |
|-----------|---------|
| Martin    | 340936  |
| Laetitia  | 325743  |
| Arthur    | 300443  |
| Amandine  | 344736  |


## Scenario
- Map = Parc
- Obstacles = arbres verts
- Robot = Chien Milou
- But : Milou retrouve Tintin, ramène un baton
- Début : On "pose un baton sur le thymio" (faire un baton en papier ou autre) -> on appuie sur le bouton start
- Obstacles globaux : arbre, eau, bancs
- Obstacles locaux : d'autres chiens ?

### Mise en place :
- Bordures : bandes de papier blanc
- Arbres : Cubes de papier cartonné vert
- Bancs (rectangles jaune)
- Eau (feuilles bleu)
- Fond : Sol d'une salle/d'un couloir (elec)
- 3x3 feuilles A3
- Goal : Carré coloré + playmobil ?
- Baton : baton imprimé en 2D qu'on scotch sur le thymio au début ?

- ATTENTION A LA LUMINOSITE / AUX REFLETS

