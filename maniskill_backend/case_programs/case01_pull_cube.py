"""Case 01 target-side LMP program for PullCube-v1."""

cube = scene.get_object("cube")
goal = scene.get_region("goal")

ret_val = robot.pull(cube, goal)
