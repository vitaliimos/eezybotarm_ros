#!/bin/sh

ros2 action send_goal /pick_and_place \
    eezybotarm_interfaces/action/PickObject \
    "{object_class: 'cup', target_pose: {header: {frame_id: 'base_link'}, pose: {position: {x: 0.33, y: 0.0, z: 0.100}, orientation: {w: 1.0}}}}" \
    --feedback
