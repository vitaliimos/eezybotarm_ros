import copy

from eezybotarm_control.workflow import run_workflow
from eezybotarm_interfaces.action import PickObject
from geometry_msgs.msg import Pose, PoseStamped
import moveit.core.planning_interface  # noqa: F401
from moveit.planning import MoveItPy, PlanningComponent, PlanRequestParameters
from moveit_msgs.msg import Constraints, MoveItErrorCodes, PositionConstraint
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

DEFAULT_FRAME = 'base_link'
END_EFFECTOR_LINK = 'ee_link'
POSITION_TOLERANCE_M = 0.003
DEFAULT_APPROACH_DISTANCE_M = 0.03
# Planning workspace as (min_x, min_y, min_z, max_x, max_y, max_z), in metres.
WORKSPACE = (-0.5, -0.5, -0.5, 0.5, 0.5, 0.5)


def _offset_z(pose: PoseStamped, dz: float) -> PoseStamped:
    """Return a copy of `pose` shifted along the Z axis by `dz` metres."""
    shifted = copy.deepcopy(pose)
    shifted.pose.position.x += dz
    shifted.pose.position.z += dz
    return shifted


def position_constraints(pose_stamped: PoseStamped) -> list[Constraints]:
    """Build a constraint: end effector must be within a small sphere around the pose."""
    sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[POSITION_TOLERANCE_M])
    sphere_pose = Pose()
    sphere_pose.position = pose_stamped.pose.position
    sphere_pose.orientation.w = 1.0

    constraint = PositionConstraint()
    constraint.header = pose_stamped.header
    if not constraint.header.frame_id:
        constraint.header.frame_id = DEFAULT_FRAME
    constraint.link_name = END_EFFECTOR_LINK
    constraint.weight = 1.0
    constraint.constraint_region.primitives.append(sphere)
    constraint.constraint_region.primitive_poses.append(sphere_pose)

    return [Constraints(name='position_only', position_constraints=[constraint])]


class PickAndPlaceActionServer(Node):
    """Action server that picks an object at `target_pose` and places it at `place_pose`."""

    def __init__(self):
        super().__init__(
            'pick_and_place_action_server',
            automatically_declare_parameters_from_overrides=True,
        )

        self.declare_parameter('approach_distance', DEFAULT_APPROACH_DISTANCE_M)
        self.approach_distance = self.get_parameter('approach_distance').value

        self.moveit = MoveItPy(node_name='pick_and_place_moveit')
        self.arm = self.moveit.get_planning_component('arm')
        self.gripper = self.moveit.get_planning_component('hand')
        self._plan_params = PlanRequestParameters(self.moveit, '')

        self._action_server = ActionServer(
            self,
            PickObject,
            'pick_and_place',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info('Pick-and-Place Action Server started!')

    def goal_callback(self, goal_request):
        position = goal_request.target_pose.pose.position
        self.get_logger().info(
            f"Goal received for '{goal_request.object_class}' "
            f'at X={position.x:.2f}, Y={position.y:.2f}'
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn('Cancellation requested!')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        pick_pose = goal_handle.request.target_pose
        place_pose = goal_handle.request.place_pose

        steps = self._build_steps(pick_pose, place_pose, self.approach_distance)
        return run_workflow(steps, goal_handle, PickObject, self.get_logger())

    def _build_steps(
        self,
        pick_pose: PoseStamped,
        place_pose: PoseStamped,
        approach_distance: float,
    ):
        pre_grasp_pose = _offset_z(pick_pose, approach_distance)
        lift_pose = _offset_z(pick_pose, approach_distance)
        pre_place_pose = _offset_z(place_pose, approach_distance)
        retreat_pose = _offset_z(place_pose, approach_distance)

        return [
            ('OPENING_GRIPPER', lambda: self._move_to_named(self.gripper, 'open')),
            ('MOVING_TO_PRE_GRASP', lambda: self._move_to_pose(pre_grasp_pose)),
            ('APPROACHING_OBJECT', lambda: self._move_to_pose(pick_pose)),
            ('CLOSING_GRIPPER', lambda: self._move_to_named(self.gripper, 'close')),
            ('LIFTING_OBJECT', lambda: self._move_to_pose(lift_pose)),
            ('MOVING_TO_PRE_PLACE', lambda: self._move_to_pose(pre_place_pose)),
            ('PLACING_OBJECT', lambda: self._move_to_pose(place_pose)),
            ('OPENING_GRIPPER', lambda: self._move_to_named(self.gripper, 'open')),
            ('RETREATING_FROM_PLACE', lambda: self._move_to_pose(retreat_pose)),
            ('MOVING_TO_HOME', lambda: self._move_to_named(self.arm, 'home')),
        ]

    def _move_to_named(self, component: PlanningComponent, name: str) -> bool:
        """Move to a named configuration from the SRDF (e.g. 'home', 'open')."""
        self._prepare(component)
        component.set_goal_state(configuration_name=name)
        return self._plan_and_execute(component)

    def _move_to_pose(self, pose_stamped: PoseStamped) -> bool:
        """Move the end effector to the given position."""
        self._prepare(self.arm)
        self.arm.set_goal_state(motion_plan_constraints=position_constraints(pose_stamped))
        return self._plan_and_execute(self.arm)

    @staticmethod
    def _prepare(component: PlanningComponent):
        component.set_start_state_to_current_state()
        component.set_workspace(*WORKSPACE)

    def _plan_and_execute(self, component: PlanningComponent) -> bool:
        response = component.plan(self._plan_params, planning_scene=None)
        if response is None:
            self.get_logger().error('Planning failed: no response')
            return False
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'Planning failed with error code {response.error_code.val}')
            return False

        if not self.moveit.execute(response.trajectory, controllers=[]):
            self.get_logger().error('Execution failed')
            return False
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceActionServer()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
