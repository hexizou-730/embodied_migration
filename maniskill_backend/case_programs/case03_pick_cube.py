"""Case 03 target-side LMP program for PickCube-v1."""

cube = scene.get_object("cube")
goal = scene.get_region("goal")

grasp_ok = robot.grasp(cube)
ret_val = robot.place(cube, goal) if grasp_ok else False
