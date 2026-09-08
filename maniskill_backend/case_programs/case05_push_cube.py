"""Case 05 target-side LMP program for PushCube-v1."""

cube = scene.get_object("cube")
goal = scene.get_region("goal")

ret_val = robot.push(cube, goal)
