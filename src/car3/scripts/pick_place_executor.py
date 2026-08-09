#!/usr/bin/env python3
import math
import time

import actionlib
import rospy
import tf.transformations as transformations
import tf2_geometry_msgs
import tf2_ros
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTolerance,
)
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import (
    PointStamped,
    PoseWithCovarianceStamped,
    PoseStamped,
)
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool, Float32, Float64, String
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5'
]


class PickPlaceExecutor:
    def __init__(self):
        self.target_category = rospy.get_param('~target_category', 'food').lower()
        self.category_to_model = rospy.get_param('~category_to_model', {})
        self.search_areas = rospy.get_param('~search_areas', {})
        self.drop_areas = rospy.get_param('~drop_areas', {})
        self.map_bounds = rospy.get_param('~map_bounds', {})
        self.arm_poses = rospy.get_param('~arm_poses', {})
        self.arm_poses_verified = bool(
            rospy.get_param('~arm_poses_verified', False))

        self.gripper_open = float(rospy.get_param('~gripper_open', 1.5))
        self.gripper_close = float(rospy.get_param('~gripper_close', 0.8))
        self.min_confidence = float(rospy.get_param('~min_confidence', 0.38))
        self.max_pose_age = float(rospy.get_param('~max_pose_age', 0.8))
        self.detection_timeout = float(
            rospy.get_param('~detection_timeout', 12.0))
        self.arm_timeout = float(rospy.get_param('~arm_timeout', 8.0))
        self.nav_timeout = float(rospy.get_param('~nav_timeout', 45.0))
        self.server_wait_timeout = float(
            rospy.get_param('~server_wait_timeout', 60.0))
        self.attach_timeout = float(rospy.get_param('~attach_timeout', 5.0))
        self.release_timeout = float(rospy.get_param('~release_timeout', 5.0))
        self.joint_tolerance = float(
            rospy.get_param('~joint_tolerance', 0.03))
        self.goal_time_tolerance = float(
            rospy.get_param('~goal_time_tolerance', 1.5))
        self.settle_position_delta = float(
            rospy.get_param('~settle_position_delta', 0.002))
        self.settle_timeout = float(rospy.get_param('~settle_timeout', 2.0))
        self.settle_samples = int(rospy.get_param('~settle_samples', 5))
        self.nav_retries = int(rospy.get_param('~nav_retries', 1))
        self.max_align_iterations = int(
            rospy.get_param('~max_align_iterations', 6))
        self.max_align_step = float(rospy.get_param('~max_align_step', 0.15))
        self.align_xy_tolerance = float(
            rospy.get_param('~align_xy_tolerance', 0.012))
        self.align_z_tolerance = float(
            rospy.get_param('~align_z_tolerance', 0.03))
        self.search_standoff = float(
            rospy.get_param('~search_standoff', 0.35))
        self.search_angles = [float(value) for value in rospy.get_param(
            '~search_angles', [0.0, 1.5708, -1.5708, 3.14159])]
        self.search_heading_offsets = [float(value) for value in rospy.get_param(
            '~search_heading_offsets', [0.35, -0.35])]
        self.drop_margin = float(rospy.get_param('~drop_margin', 0.05))
        self.drop_settle_samples = int(
            rospy.get_param('~drop_settle_samples', 5))
        self.drop_settle_delta = float(
            rospy.get_param('~drop_settle_delta', 0.01))
        self.navigation_ready_timeout = float(
            rospy.get_param('~navigation_ready_timeout', 60.0))
        self.navigation_startup_delay = float(
            rospy.get_param('~navigation_startup_delay', 5.0))
        self.navigation_stable_time = float(
            rospy.get_param('~navigation_stable_time', 3.0))
        self.navigation_pose_tolerance = float(
            rospy.get_param('~navigation_pose_tolerance', 0.03))
        self.navigation_data_timeout = float(
            rospy.get_param('~navigation_data_timeout', 2.5))
        self.navigation_min_costmap_updates = int(
            rospy.get_param('~navigation_min_costmap_updates', 3))

        self.joint_values = {}
        self.map_grid = None
        self.scan_message = None
        self.global_costmap = None
        self.local_costmap = None
        self.amcl_pose = None
        self._map_wall_time = None
        self._scan_wall_time = None
        self._global_costmap_wall_time = None
        self._local_costmap_wall_time = None
        self._amcl_wall_time = None
        self._global_costmap_updates = 0
        self._local_costmap_updates = 0
        self._readiness_subscribers = []
        self.category = 'unknown'
        self.confidence = 0.0
        self.vision_pose = None
        self.vision_pose_received = rospy.Time(0)
        self.ready = False
        self.grasp_state = 'IDLE'
        self.attached_model = ''
        self.attach_offset = None
        self.grasp_tcp_in_base = None
        self.transport_tcp_in_base = None
        self.place_tcp_in_base = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.nav_client = actionlib.SimpleActionClient(
            '/move_base', MoveBaseAction)
        self.arm_client = actionlib.SimpleActionClient(
            '/arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        self.clear_costmaps = rospy.ServiceProxy(
            '/move_base/clear_costmaps', Empty)
        self.make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        self.get_model_state = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)

        self.gripper_pub = rospy.Publisher(
            '/gripper_controller/command', Float64, queue_size=1)
        self.state_pub = rospy.Publisher(
            '~state', String, queue_size=1, latch=True)
        self.result_pub = rospy.Publisher(
            '~result', String, queue_size=1, latch=True)

        rospy.Subscriber('/joint_states', JointState, self._joint_cb)
        self._readiness_subscribers.extend([
            rospy.Subscriber('/map', OccupancyGrid, self._map_cb),
            rospy.Subscriber('/scan', LaserScan, self._scan_cb),
            rospy.Subscriber(
                '/move_base/global_costmap/costmap', OccupancyGrid,
                self._global_costmap_cb),
            rospy.Subscriber(
                '/move_base/local_costmap/costmap', OccupancyGrid,
                self._local_costmap_cb),
            rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped,
                             self._amcl_pose_cb),
        ])
        rospy.Subscriber('/cube_vision/category', String, self._category_cb)
        rospy.Subscriber('/cube_vision/confidence', Float32, self._confidence_cb)
        rospy.Subscriber('/cube_vision/pose', PoseStamped, self._vision_pose_cb)
        rospy.Subscriber('/grasp_attach/ready', Bool, self._ready_cb)
        rospy.Subscriber('/grasp_attach/state', String, self._grasp_state_cb)
        rospy.Subscriber(
            '/grasp_attach/attached_model', String, self._attached_model_cb)

        rospy.on_shutdown(self._cancel_actions)
        self._validate_config()

    def _validate_config(self):
        if self.target_category not in self.category_to_model:
            raise rospy.ROSInitException(
                'target_category has no model mapping: ' + self.target_category)
        if self.target_category not in self.drop_areas:
            raise rospy.ROSInitException(
                'target_category has no drop area: ' + self.target_category)
        if not self.search_areas:
            raise rospy.ROSInitException('search_areas is empty')
        for name in ('navigation', 'observe', 'grasp', 'transport', 'place'):
            self._read_arm_pose(name)
        for name, value in (
                ('gripper_open', self.gripper_open),
                ('gripper_close', self.gripper_close)):
            if not math.isfinite(value) or not -1.51 <= value <= 1.51:
                raise rospy.ROSInitException(name + ' is outside r_joint limits')
        required_bounds = ('x_min', 'x_max', 'y_min', 'y_max')
        if any(key not in self.map_bounds for key in required_bounds):
            raise rospy.ROSInitException('map_bounds is incomplete')

    def _read_arm_pose(self, name):
        pose = self.arm_poses.get(name)
        if not isinstance(pose, dict):
            raise rospy.ROSInitException('missing arm pose: ' + name)
        positions = pose.get('positions')
        duration = pose.get('duration')
        if not isinstance(positions, list) or len(positions) != len(JOINT_NAMES):
            raise rospy.ROSInitException(name + ' must contain five positions')
        if duration is None or float(duration) <= 0.0:
            raise rospy.ROSInitException(name + ' duration must be positive')
        positions = [float(value) for value in positions]
        if not all(math.isfinite(value) and -3.14 <= value <= 3.14
                   for value in positions):
            raise rospy.ROSInitException(name + ' contains invalid joint values')
        return positions, float(duration)

    def _joint_cb(self, msg):
        values = dict(zip(msg.name, msg.position))
        self.joint_values.update(values)

    def _map_cb(self, msg):
        self.map_grid = msg
        self._map_wall_time = time.monotonic()

    def _scan_cb(self, msg):
        self.scan_message = msg
        self._scan_wall_time = time.monotonic()

    def _global_costmap_cb(self, msg):
        self.global_costmap = msg
        self._global_costmap_wall_time = time.monotonic()
        self._global_costmap_updates += 1

    def _local_costmap_cb(self, msg):
        self.local_costmap = msg
        self._local_costmap_wall_time = time.monotonic()
        self._local_costmap_updates += 1

    def _amcl_pose_cb(self, msg):
        self.amcl_pose = msg
        self._amcl_wall_time = time.monotonic()

    def _category_cb(self, msg):
        self.category = msg.data.lower()

    def _confidence_cb(self, msg):
        self.confidence = float(msg.data)

    def _vision_pose_cb(self, msg):
        self.vision_pose = msg
        self.vision_pose_received = rospy.Time.now()

    def _ready_cb(self, msg):
        self.ready = bool(msg.data)

    def _grasp_state_cb(self, msg):
        self.grasp_state = msg.data

    def _attached_model_cb(self, msg):
        self.attached_model = msg.data

    def _set_state(self, state):
        rospy.loginfo('pick_place state: %s', state)
        self.state_pub.publish(String(data=state))

    def _cancel_actions(self):
        self.nav_client.cancel_all_goals()
        self.arm_client.cancel_all_goals()

    def _wait_for_action_server(self, client, label):
        # actionlib's timeout uses simulated ROS time. During Gazebo startup
        # that clock can be paused, so inspect the negotiated ROS connections
        # directly and use wall time for this startup-only wait.
        action_client = client.action_client
        deadline = time.monotonic() + self.server_wait_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            status = action_client.last_status_msg
            if status:
                server_id = status._connection_header.get('callerid')
                goal_connected = action_client.pub_goal.impl.has_connection(
                    server_id)
                cancel_connected = action_client.pub_cancel.impl.has_connection(
                    server_id)
                result_connected = any(
                    connection.callerid_pub == server_id
                    for connection in action_client.result_sub.impl.connections)
                feedback_connected = any(
                    connection.callerid_pub == server_id
                    for connection in action_client.feedback_sub.impl.connections)
                if (goal_connected and cancel_connected and result_connected
                        and feedback_connected):
                    return True
            rospy.loginfo_throttle(5.0, 'waiting for %s action server', label)
            time.sleep(0.05)
        return False

    def _wait_joint_state(self, timeout):
        deadline = time.monotonic() + float(timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if all(name in self.joint_values for name in JOINT_NAMES):
                return True
            time.sleep(0.05)
        return False

    @staticmethod
    def _valid_grid(grid):
        return (grid is not None and grid.info.width > 0
                and grid.info.height > 0
                and len(grid.data) == grid.info.width * grid.info.height)

    def _navigation_readiness_missing(self):
        now = time.monotonic()
        missing = []
        if not self._valid_grid(self.map_grid):
            missing.append('map')
        if (self.scan_message is None or not self.scan_message.ranges
                or not math.isfinite(self.scan_message.angle_min)
                or not math.isfinite(self.scan_message.angle_increment)
                or self._scan_wall_time is None
                or now - self._scan_wall_time > self.navigation_data_timeout):
            missing.append('scan')
        if not self._valid_grid(self.global_costmap):
            rospy.logdebug_throttle(5.0, 'global costmap message not observed yet')
        if not self._valid_grid(self.local_costmap):
            rospy.logdebug_throttle(5.0, 'local costmap message not observed yet')
        if self.amcl_pose is None:
            rospy.logdebug_throttle(5.0, 'amcl_pose message not observed yet')
        try:
            self._base_pose_map()
        except RuntimeError:
            missing.append('map_to_base_tf')
        return missing

    def _wait_for_navigation_ready(self):
        self._set_state('WAIT_NAVIGATION_READY')
        rospy.loginfo(
            'waiting %.1f seconds for navigation startup to settle',
            self.navigation_startup_delay)
        time.sleep(self.navigation_startup_delay)
        deadline = time.monotonic() + self.navigation_ready_timeout
        stable_since = None
        previous_pose = None
        last_missing = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            missing = self._navigation_readiness_missing()
            if missing != last_missing:
                rospy.loginfo(
                    'navigation readiness waiting for: %s',
                    ', '.join(missing) if missing else 'pose stability')
                last_missing = missing
            if not missing:
                try:
                    current_pose = self._base_pose_map()
                except RuntimeError:
                    current_pose = None
                if current_pose is not None:
                    if previous_pose is None:
                        stable_since = time.monotonic()
                    else:
                        position_delta = math.hypot(
                            current_pose[0] - previous_pose[0],
                            current_pose[1] - previous_pose[1])
                        yaw_delta = abs(math.atan2(
                            math.sin(current_pose[2] - previous_pose[2]),
                            math.cos(current_pose[2] - previous_pose[2])))
                        if (position_delta > self.navigation_pose_tolerance
                                or yaw_delta > self.navigation_pose_tolerance):
                            stable_since = time.monotonic()
                    previous_pose = current_pose
                    if (stable_since is not None
                            and time.monotonic() - stable_since
                            >= self.navigation_stable_time):
                        rospy.loginfo('navigation stack is ready')
                        return
            else:
                stable_since = None
                previous_pose = None
            time.sleep(0.05)
        raise RuntimeError(
            'navigation stack not ready: {}'.format(
                ', '.join(last_missing or ['pose stability'])))

    def _move_arm(self, name):
        positions, duration = self._read_arm_pose(name)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.05)
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = rospy.Duration(self.goal_time_tolerance)
        goal.goal_tolerance = [
            JointTolerance(name=joint, position=self.joint_tolerance,
                           velocity=-1.0, acceleration=-1.0)
            for joint in JOINT_NAMES
        ]
        goal.path_tolerance = [
            JointTolerance(name=joint, position=-1.0,
                           velocity=-1.0, acceleration=-1.0)
            for joint in JOINT_NAMES
        ]

        self.arm_client.send_goal(goal)
        if not self.arm_client.wait_for_result(rospy.Duration(self.arm_timeout)):
            self.arm_client.cancel_goal()
            raise RuntimeError(name + ' arm action timeout')
        action_state = self.arm_client.get_state()
        if action_state != GoalStatus.SUCCEEDED:
            result = self.arm_client.get_result()
            error_code = getattr(result, 'error_code', 'unknown')
            if error_code not in (-4, -5):
                raise RuntimeError(
                    '{} arm action failed: state={}, error_code={}'.format(
                        name, action_state, error_code))

        deadline = rospy.Time.now() + rospy.Duration(self.settle_timeout)
        settled = 0
        previous = None
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            current = [self.joint_values.get(joint) for joint in JOINT_NAMES]
            if any(value is None for value in current):
                rospy.sleep(0.05)
                continue
            position_error = max(
                abs(value - target) for value, target in zip(current, positions))
            position_delta = (float('inf') if previous is None else max(
                abs(value - old) for value, old in zip(current, previous)))
            if (position_error <= self.joint_tolerance
                    and position_delta <= self.settle_position_delta):
                settled += 1
                if settled >= self.settle_samples:
                    return
            else:
                settled = 0
            previous = current
            rospy.sleep(0.05)
        raise RuntimeError(name + ' arm pose did not settle')

    def _command_gripper(self, target, label):
        self.gripper_pub.publish(Float64(data=target))
        deadline = rospy.Time.now() + rospy.Duration(4.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            position = self.joint_values.get('r_joint')
            if position is not None and abs(position - target) <= 0.04:
                return
            rospy.sleep(0.05)
        raise RuntimeError(label + ' gripper position timeout')

    def _lookup_tcp_in_base(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint', 'tcp_link', rospy.Time(0),
                rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            raise RuntimeError('tcp_link TF unavailable: {}'.format(exc))
        return (transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z)

    def _measure_arm_geometry(self):
        self._set_state('MEASURE_ARM_GEOMETRY')
        self._move_arm('grasp')
        self.grasp_tcp_in_base = self._lookup_tcp_in_base()
        self._move_arm('place')
        self.place_tcp_in_base = self._lookup_tcp_in_base()
        self._move_arm('transport')
        self.transport_tcp_in_base = self._lookup_tcp_in_base()
        rospy.loginfo(
            'TF-derived tcp positions: grasp=(%.4f, %.4f, %.4f), '
            'transport=(%.4f, %.4f, %.4f), place=(%.4f, %.4f, %.4f)',
            *(self.grasp_tcp_in_base + self.transport_tcp_in_base
              + self.place_tcp_in_base))
        self._move_arm('navigation')

    def _fresh_detection(self, since_received=None, timeout=None):
        timeout = self.detection_timeout if timeout is None else timeout
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            pose = self.vision_pose
            received = self.vision_pose_received
            if pose is not None and self.category == self.target_category:
                stamp = pose.header.stamp
                age = ((rospy.Time.now() - stamp).to_sec()
                       if stamp != rospy.Time(0) else 0.0)
                received_after = (since_received is None
                                  or received > since_received)
                stamp_is_usable = (stamp == rospy.Time(0)
                                   or age >= 0.0)
                if (received_after and pose.header.frame_id
                        and stamp_is_usable and age <= self.max_pose_age
                        and self.confidence >= self.min_confidence):
                    return pose
            rospy.sleep(0.05)
        return None

    def _point_in_frame(self, pose, target_frame):
        point = PointStamped()
        point.header = pose.header
        point.point = pose.pose.position
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, pose.header.frame_id, pose.header.stamp,
                rospy.Duration(0.5))
            return tf2_geometry_msgs.do_transform_point(point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            raise RuntimeError('vision pose TF unavailable: {}'.format(exc))

    def _base_pose_map(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rospy.Time(0), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            raise RuntimeError('map to base TF unavailable: {}'.format(exc))
        q = transform.transform.rotation
        yaw = transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])[2]
        return (transform.transform.translation.x,
                transform.transform.translation.y, yaw)

    def _inside_map(self, x, y, margin=0.0):
        margin = float(margin)
        return (float(self.map_bounds['x_min']) + margin <= x
                <= float(self.map_bounds['x_max']) - margin
                and float(self.map_bounds['y_min']) + margin <= y
                <= float(self.map_bounds['y_max']) - margin)

    @staticmethod
    def _move_base_goal(x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time(0)
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        quaternion = transformations.quaternion_from_euler(0.0, 0.0, yaw)
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        return goal

    def _wait_for_plan(self, x, y, yaw, timeout):
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                start_x, start_y, start_yaw = self._base_pose_map()
                start = PoseStamped()
                start.header.frame_id = 'map'
                start.header.stamp = rospy.Time(0)
                start.pose.position.x = start_x
                start.pose.position.y = start_y
                start_q = transformations.quaternion_from_euler(
                    0.0, 0.0, start_yaw)
                start.pose.orientation.x = start_q[0]
                start.pose.orientation.y = start_q[1]
                start.pose.orientation.z = start_q[2]
                start.pose.orientation.w = start_q[3]
                goal = PoseStamped()
                goal.header.frame_id = 'map'
                goal.header.stamp = rospy.Time(0)
                goal.pose.position.x = x
                goal.pose.position.y = y
                goal_q = transformations.quaternion_from_euler(0.0, 0.0, yaw)
                goal.pose.orientation.x = goal_q[0]
                goal.pose.orientation.y = goal_q[1]
                goal.pose.orientation.z = goal_q[2]
                goal.pose.orientation.w = goal_q[3]
                response = self.make_plan(start, goal, 0.0)
                if response.plan.poses:
                    return True
            except (rospy.ServiceException, RuntimeError) as exc:
                rospy.loginfo_throttle(
                    5.0, 'waiting for move_base plan service/TF: %s', exc)
            rospy.loginfo_throttle(
                5.0, 'waiting for a non-empty move_base plan')
            time.sleep(0.2)
        return False

    def _navigate(self, x, y, yaw, label, retries=None):
        if not self._inside_map(x, y):
            rospy.logwarn('%s goal is outside map: (%.3f, %.3f)', label, x, y)
            return False
        retries = self.nav_retries if retries is None else retries
        for attempt in range(retries + 1):
            if not self._wait_for_plan(x, y, yaw, self.nav_timeout):
                rospy.logwarn('%s did not get a plan before timeout', label)
                return False
            self.nav_client.send_goal(self._move_base_goal(x, y, yaw))
            deadline = time.monotonic() + self.nav_timeout
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if (label.startswith('drop approach')
                        and (self.grasp_state != 'GRASPING'
                             or self.attached_model != self.category_to_model[
                                 self.target_category])):
                    self.nav_client.cancel_goal()
                    raise RuntimeError('attachment lost during transport')
                state = self.nav_client.get_state()
                if state == GoalStatus.SUCCEEDED:
                    return True
                if state in (GoalStatus.ABORTED, GoalStatus.REJECTED,
                             GoalStatus.PREEMPTED, GoalStatus.RECALLED,
                             GoalStatus.LOST):
                    break
                time.sleep(0.05)
            self.nav_client.cancel_goal()
            if attempt < retries:
                try:
                    self.clear_costmaps()
                except rospy.ServiceException as exc:
                    rospy.logwarn('clear_costmaps failed: %s', exc)
        rospy.logwarn('%s navigation failed', label)
        return False

    def _search(self):
        self._set_state('SEARCH_SOURCE')
        # The first coarse goal uses the model's default arm posture. After an
        # observation, every further base motion first restores navigation.
        for area_name in sorted(self.search_areas):
            area = self.search_areas[area_name]
            center_x = 0.5 * (float(area['x_min']) + float(area['x_max']))
            center_y = 0.5 * (float(area['y_min']) + float(area['y_max']))
            for index, angle in enumerate(self.search_angles):
                goal_x = center_x - self.search_standoff * math.cos(angle)
                goal_y = center_y - self.search_standoff * math.sin(angle)
                if not self._navigate(
                        goal_x, goal_y, angle,
                        '{} observation {}'.format(area_name, index), retries=0):
                    continue
                for heading_offset in [0.0] + self.search_heading_offsets:
                    if heading_offset:
                        self._move_arm('navigation')
                        _, _, current_yaw = self._base_pose_map()
                        if not self._navigate(
                                goal_x, goal_y, current_yaw + heading_offset,
                                '{} heading adjustment'.format(area_name),
                                retries=0):
                            continue
                    self._move_arm('observe')
                    since_received = self.vision_pose_received
                    pose = self._fresh_detection(since_received)
                    if pose is not None:
                        rospy.loginfo(
                            'found %s in %s', self.target_category, area_name)
                        return pose
                    self._move_arm('navigation')
        raise RuntimeError('target was not detected in any search area')

    def _align_to_grasp(self, first_pose):
        self._set_state('ALIGN_TO_GRASP')
        # Vision acquisition happened with the arm lowered and the base parked.
        # Raise the arm before any base motion so it cannot enter the laser scan
        # plane; the grasp posture is restored only after alignment converges.
        self._move_arm('navigation')
        target = self._point_in_frame(first_pose, 'map').point
        desired_x, desired_y, _ = self.grasp_tcp_in_base
        for iteration in range(self.max_align_iterations + 1):
            base_x, base_y, base_yaw = self._base_pose_map()
            cos_yaw = math.cos(base_yaw)
            sin_yaw = math.sin(base_yaw)
            local_x = cos_yaw * (target.x - base_x) + sin_yaw * (target.y - base_y)
            local_y = -sin_yaw * (target.x - base_x) + cos_yaw * (target.y - base_y)
            error_x = local_x - desired_x
            error_y = local_y - desired_y
            error_xy = math.hypot(error_x, error_y)
            rospy.loginfo(
                'alignment %d: cube_base=(%.3f, %.3f) error=(%.3f, %.3f)',
                iteration, local_x, local_y, error_x, error_y)
            if error_xy <= self.align_xy_tolerance:
                return
            if iteration >= self.max_align_iterations:
                break

            scale = min(1.0, self.max_align_step / error_xy)
            goal_x = base_x + scale * (target.x - base_x)
            goal_y = base_y + scale * (target.y - base_y)
            if not self._navigate(
                    goal_x, goal_y, base_yaw,
                    'visual alignment {}'.format(iteration), retries=0):
                raise RuntimeError('visual alignment navigation failed')
        raise RuntimeError('visual alignment did not converge')

    def _prepare_stationary_grasp(self):
        self._set_state('PREPARE_GRASP')
        self._move_arm('grasp')
        deadline = rospy.Time.now() + rospy.Duration(3.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.ready:
                return
            rospy.sleep(0.05)
        raise RuntimeError('object is not inside the tcp grasp box after alignment')

    def _wait_attach(self):
        expected = self.category_to_model[self.target_category]
        deadline = rospy.Time.now() + rospy.Duration(self.attach_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.grasp_state == 'GRASPING' and self.attached_model == expected:
                return
            rospy.sleep(0.05)
        raise RuntimeError(
            'attach timeout: ready={}, state={}, model={}'.format(
                self.ready, self.grasp_state, self.attached_model))

    def _check_attachment(self):
        expected = self.category_to_model[self.target_category]
        if self.grasp_state != 'GRASPING' or self.attached_model != expected:
            raise RuntimeError('attachment lost during transport')

    def _grasp(self):
        self._set_state('GRASP')
        # _prepare_stationary_grasp has already positioned the arm while the
        # base is stopped; only the gripper moves in this state.
        if not self.ready:
            raise RuntimeError('object is not inside the tcp grasp box')
        self._command_gripper(self.gripper_close, 'close')
        self._wait_attach()

    def _navigate_to_drop(self):
        self._set_state('NAVIGATE_TO_DROP')
        # Change to the verified placement posture before computing the
        # placement TCP offset. The attached model follows tcp_link while the
        # base is transported.
        self._move_arm('transport')
        # The base goal is computed for the placement posture, but navigation
        # itself stays in the raised transport posture.
        area = self.drop_areas[self.target_category]
        drop_x = 0.5 * (float(area['x_min']) + float(area['x_max']))
        drop_y = 0.5 * (float(area['y_min']) + float(area['y_max']))
        _, _, current_yaw = self._base_pose_map()
        headings = [current_yaw, 0.0, math.pi / 2.0, -math.pi / 2.0, math.pi]
        tcp_x, tcp_y, _ = self.place_tcp_in_base
        for index, yaw in enumerate(headings):
            goal_x = drop_x - (math.cos(yaw) * tcp_x - math.sin(yaw) * tcp_y)
            goal_y = drop_y - (math.sin(yaw) * tcp_x + math.cos(yaw) * tcp_y)
            if self._navigate(
                    goal_x, goal_y, yaw,
                    'drop approach {}'.format(index), retries=0):
                self._check_attachment()
                return
        raise RuntimeError('no drop approach goal was reachable')

    def _wait_release(self):
        deadline = rospy.Time.now() + rospy.Duration(self.release_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.grasp_state == 'IDLE' and not self.attached_model:
                return
            rospy.sleep(0.05)
        raise RuntimeError('object did not release')

    def _world_point_to_map(self, point_world):
        # Gazebo exposes model states in its world frame, while task areas are
        # map coordinates. Derive the planar world->map transform from the
        # robot pose instead of assuming both frames share an origin.
        try:
            response = self.get_model_state('car3', 'world')
        except rospy.ServiceException as exc:
            raise RuntimeError('car3 world state unavailable: {}'.format(exc))
        if not response.success:
            raise RuntimeError('car3 world state unavailable: ' + response.status_message)
        robot_world = response.pose
        robot_q = [robot_world.orientation.x, robot_world.orientation.y,
                   robot_world.orientation.z, robot_world.orientation.w]
        robot_yaw = transformations.euler_from_quaternion(robot_q)[2]
        base_x, base_y, base_yaw = self._base_pose_map()
        dx = point_world.x - robot_world.position.x
        dy = point_world.y - robot_world.position.y
        cos_world = math.cos(robot_yaw)
        sin_world = math.sin(robot_yaw)
        local_x = cos_world * dx + sin_world * dy
        local_y = -sin_world * dx + cos_world * dy
        cos_map = math.cos(base_yaw)
        sin_map = math.sin(base_yaw)
        return (base_x + cos_map * local_x - sin_map * local_y,
                base_y + sin_map * local_x + cos_map * local_y,
                point_world.z)

    def _verify_drop(self):
        model = self.category_to_model[self.target_category]
        area = self.drop_areas[self.target_category]
        x_min = float(area['x_min']) + self.drop_margin
        x_max = float(area['x_max']) - self.drop_margin
        y_min = float(area['y_min']) + self.drop_margin
        y_max = float(area['y_max']) - self.drop_margin
        z_min = float(rospy.get_param('~drop_z_min', 0.0))
        z_max = float(rospy.get_param('~drop_z_max', 0.12))
        previous = None
        settled = 0
        deadline = rospy.Time.now() + rospy.Duration(5.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                response = self.get_model_state(model, 'world')
            except rospy.ServiceException as exc:
                raise RuntimeError('get_model_state failed: {}'.format(exc))
            if not response.success:
                raise RuntimeError('model state unavailable: ' + response.status_message)
            map_x, map_y, map_z = self._world_point_to_map(response.pose.position)
            if not (x_min <= map_x <= x_max and y_min <= map_y <= y_max
                    and z_min <= map_z <= z_max):
                settled = 0
                previous = None
                rospy.sleep(0.2)
                continue
            if previous is not None:
                delta = math.hypot(map_x - previous[0], map_y - previous[1])
                settled = settled + 1 if delta <= self.drop_settle_delta else 0
                if settled >= self.drop_settle_samples:
                    return
            previous = (map_x, map_y)
            rospy.sleep(0.2)
        raise RuntimeError('released model is not stable inside the drop area')

    def _place(self):
        self._set_state('PLACE')
        self._move_arm('place')
        self._command_gripper(self.gripper_open, 'open')
        self._wait_release()
        self._verify_drop()

    def run(self):
        self._set_state('INIT_VALIDATE')
        if not self.arm_poses_verified:
            raise RuntimeError(
                'arm_poses_verified is false; confirm the three arm poses are safe')
        rospy.loginfo('waiting for move_base action server')
        if not self._wait_for_action_server(self.nav_client, 'move_base'):
            raise RuntimeError('move_base action server unavailable')
        rospy.loginfo('move_base action server is ready')
        rospy.loginfo('waiting for arm trajectory action server')
        if not self._wait_for_action_server(
                self.arm_client, 'arm trajectory'):
            raise RuntimeError('arm trajectory action server unavailable')
        rospy.loginfo('arm trajectory action server is ready')
        if not self._wait_joint_state(8.0):
            raise RuntimeError('joint state unavailable')
        rospy.wait_for_service('/gazebo/get_model_state', timeout=10.0)

        self._set_state('RESET_GRIPPER')
        self._command_gripper(self.gripper_open, 'open')
        deadline = rospy.Time.now() + rospy.Duration(2.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.grasp_state == 'IDLE' and not self.attached_model:
                break
            rospy.sleep(0.05)
        if self.grasp_state != 'IDLE' or self.attached_model:
            raise RuntimeError('grasp backend is not idle')

        # Keep the model's default arm posture for the first coarse navigation.
        # Later base motions still raise the arm after observe/grasp/place.
        self._wait_for_navigation_ready()

        # Search first. Geometry measurement moves through grasp/place poses
        # and must not delay the initial navigation.
        detection = self._search()
        self._measure_arm_geometry()
        self._align_to_grasp(detection)
        self._prepare_stationary_grasp()
        self._grasp()
        self._move_arm('transport')
        self._navigate_to_drop()
        self._place()
        self._set_state('SUCCESS')
        self.result_pub.publish(String(data='SUCCESS:' + self.target_category))
        rospy.loginfo('pick-place succeeded for %s', self.target_category)


def main():
    rospy.init_node('pick_place_executor')
    executor = None
    try:
        executor = PickPlaceExecutor()
        if not bool(rospy.get_param('~start_task', False)):
            executor._set_state('WAITING_TO_START')
            rospy.loginfo(
                'Set ~start_task=true and relaunch to run the single-object task')
            rospy.spin()
            return
        executor.run()
    except (rospy.ROSException, rospy.ROSInitException, RuntimeError) as exc:
        rospy.logerr('pick-place FAILED: %s', exc)
        if executor is not None:
            executor._set_state('FAILED')
            executor.result_pub.publish(String(data='FAILED:' + str(exc)))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
