# Vendored ManiSkill reference code

`mani_skill_motionplanning/` contains the motion-planning reference code
distributed with ManiSkill 3.0.1. It is kept locally so the frozen source
programs do not depend on an untracked checkout of the ManiSkill examples.
Only the shared Panda planning helpers are included; the eleven task solutions
are inlined in `source_programs/candidates/`. The experiment remains pinned to
`mani_skill==3.0.1`.
