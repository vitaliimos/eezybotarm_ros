#!/bin/sh

ros2 action send_goal /pick_and_place \
    eezybotarm_interfaces/action/PickObject \
    "{object_class: 'white_cube', \
      target_pose: {header: {frame_id: 'base_link'}, pose: {position: {x: 0.22, y: 0.0, z: 0.1025}, orientation: {w: 1.0}}}, \
      place_pose: {header: {frame_id: 'base_link'}, pose: {position: {x: 0.22, y: -0.09, z: 0.1025}, orientation: {w: 1.0}}}}" \
    --feedback
