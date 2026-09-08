"""Frozen multi-object program shared by Panda and Fetch."""

red = scene.get_object("cubeA")
green = scene.get_object("cubeB")
blue = scene.get_object("cubeC")

base_ok = robot.prepare_base(red, green)
ret_val = robot.stack_on(blue, red, green) if base_ok else False
