# TODO

## Gripper `rad_min`/`rad_max` used as travel limits

`src/eezybotarm_moveit_config/config/eezybotarm.ros2_control.xacro` configures
`finger_left_joint` with `rad_min=-0.15` / `rad_max=0.65` for `us_min=600` / `us_max=2400`.

This breaks the rule in `AGENTS.md`: `rad_min`/`rad_max` are the actuator's calibration
scale (±1.5708 rad for 600–2400 µs on an MG995), not joint limits. Squeezing them to
0.8 rad stretches the PWM scale about 3.9x, so a small joint command moves the servo
too far and it can stall.

Fix:
- Set the gripper `rad_min`/`rad_max` to `-1.5708` / `1.5708`, and add a `gear_ratio`
  if the finger linkage isn't 1:1.
- Make sure the finger travel limits live in `robot.urdf` and `joint_limits.yaml`.
- Re-check the gripper open/close positions on hardware.
