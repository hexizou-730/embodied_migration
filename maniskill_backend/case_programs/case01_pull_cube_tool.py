"""Case 01 target-side LMP program.

The full-stack LLM runner may migrate this target program without changing the
validated Panda source program in ``tasks.py``.
"""

tool = scene.get_object("l_shape_tool")
cube = scene.get_object("cube")
workspace = scene.get_region("workspace")

ok = robot.hook_object(tool, cube)
if ok:
    ret_val = robot.pull_with_tool(tool, cube, workspace)
else:
    ret_val = "failure: hook"
