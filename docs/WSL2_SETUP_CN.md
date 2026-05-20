# WSL2 Development Note

WSL2 can still be used for editing, unit tests, prompt debugging, and log
inspection. It is not the final platform for the real ManiSkill experiments in
this project.

Use the Polytechnique GPU machines or native Ubuntu with NVIDIA Vulkan support
for real simulator runs.

Recommended WSL2 workflow:

```bash
cd ~/Embodied/embodied_migration
conda activate em-ms
python -m unittest discover -s tests -v
```

For real simulation commands, see `docs/RUN.md`.
