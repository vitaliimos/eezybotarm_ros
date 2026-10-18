import time

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


class PickAndPlaceActionServer(Node):

    def __init__(self):
        super().__init__(
            'pick_and_place_action_server',
            automatically_declare_parameters_from_overrides=True,
        )

        self.moveit = MoveItPy(node_name='pick_and_place_moveit')
        self.arm = self.moveit.get_planning_component('arm')
        self.gripper = self.moveit.get_planning_component('hand')

        self._plan_params = PlanRequestParameters(self.moveit, '')

        self._callback_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            PickObject,
            'pick_and_place',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._callback_group,
        )

        self.get_logger().info('Pick-and-Place Action Server started!')

    def goal_callback(self, goal_request):
        self.get_logger().info(
            f"Goal received for object '{goal_request.object_class}' "
            f'at X={goal_request.target_pose.pose.position.x:.2f}, '
            f'Y={goal_request.target_pose.pose.position.y:.2f}'
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn('Cancellation requested!')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Pick-and-Place started...')

        started_at = time.monotonic()
        target_pose = goal_handle.request.target_pose
        place_pose = goal_handle.request.place_pose
        feedback_msg = PickObject.Feedback()
        result = PickObject.Result()

        def fail(error_code: str):
            goal_handle.abort()
            result.success = False
            result.execution_time = time.monotonic() - started_at
            result.error_code = error_code
            return result

        feedback_msg.current_phase = 'OPENING_GRIPPER'
        feedback_msg.progress_percentage = 0.0
        goal_handle.publish_feedback(feedback_msg)
        if not self._plan_and_execute(self.gripper, configuration_name='open'):
            return fail('Failed to open gripper')

        feedback_msg.current_phase = 'MOVING_TO_PRE_GRASP'
        feedback_msg.progress_percentage = 25.0
        goal_handle.publish_feedback(feedback_msg)
        if not self._plan_and_execute(
            self.arm,
            constraints=self._position_only_constraints(target_pose, 'ee_link'),
        ):
            return fail('Failed to move arm to target pose')

        feedback_msg.current_phase = 'CLOSING_GRIPPER'
        feedback_msg.progress_percentage = 50.0
        goal_handle.publish_feedback(feedback_msg)
        if not self._plan_and_execute(self.gripper, configuration_name='close'):
            return fail('Failed to close gripper')

        feedback_msg.current_phase = 'MOVING_TO_POST_GRASP'
        feedback_msg.progress_percentage = 75.0
        goal_handle.publish_feedback(feedback_msg)
        if not self._plan_and_execute(self.arm, configuration_name='home'):
            return fail('Failed to move arm to post-grasp pose')

        feedback_msg.current_phase = 'MOVING_TO_PLACE_POSITION'
        feedback_msg.progress_percentage = 80.0
        goal_handle.publish_feedback(feedback_msg)
        if not self._plan_and_execute(
            self.arm,
            constraints=self._position_only_constraints(place_pose, 'ee_link'),
        ):
            return fail('Failed to move arm to place pose')

        feedback_msg.current_phase = 'OPENING_GRIPPER'
        feedback_msg.progress_percentage = 0.0
        goal_handle.publish_feedback(feedback_msg)
        if not self._plan_and_execute(self.gripper, configuration_name='open'):
            return fail('Failed to open gripper')

        feedback_msg.current_phase = 'COMPLETE'
        feedback_msg.progress_percentage = 100.0
        goal_handle.publish_feedback(feedback_msg)
        goal_handle.succeed()
        result.success = True
        result.execution_time = time.monotonic() - started_at
        result.error_code = ''
        return result

    @staticmethod
    def _position_only_constraints(
        pose_stamped: PoseStamped,
        link: str,
        tolerance_m: float = 0.003,
    ) -> list:
        pc = PositionConstraint()
        pc.header = pose_stamped.header
        if not pc.header.frame_id:
            pc.header.frame_id = 'base_link'
        pc.weight = 1.0
        pc.link_name = link
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [tolerance_m]
        region_pose = Pose()
        region_pose.position = pose_stamped.pose.position
        region_pose.orientation.w = 1.0
        pc.constraint_region.primitives.append(sphere)
        pc.constraint_region.primitive_poses.append(region_pose)
        constraints = Constraints(name='position_only')
        constraints.position_constraints.append(pc)
        return [constraints]

    def _plan_and_execute(
        self,
        component: PlanningComponent,
        configuration_name: str = None,
        pose_stamped: PoseStamped = None,
        pose_link: str = None,
        constraints: list = None,
    ) -> bool:
        component.set_start_state_to_current_state()
        component.set_workspace(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5)

        if configuration_name:
            component.set_goal_state(configuration_name=configuration_name)
        elif constraints:
            component.set_goal_state(motion_plan_constraints=constraints)
        elif pose_stamped and pose_link:
            component.set_goal_state(pose_stamped_msg=pose_stamped, pose_link=pose_link)
        else:
            self.get_logger().error('No goal specified for plan_and_execute')
            return False

        response = component.plan(self._plan_params, planning_scene=None)
        if response is None or response.error_code.val != MoveItErrorCodes.SUCCESS:
            err_val = response.error_code.val if response else 'None'
            self.get_logger().error(f'Planning failed with error code {err_val}')
            return False

        success = self.moveit.execute(response.trajectory, controllers=[])
        if not success:
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
