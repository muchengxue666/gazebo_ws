#!/usr/bin/env python3
import collections
import math
import time

import actionlib
import numpy as np
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
from geometry_msgs.msg import (
    Point,
    PointStamped,
    PoseWithCovarianceStamped,
    PoseStamped,
    Twist,
)
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Bool, Float32, Float64, String
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5'
]
TARGET_CATEGORIES = ('food', 'daily', 'electronics')
MAX_FAILURE_RECOVERY_ATTEMPTS = 3


class GraspPreparationError(RuntimeError):
    pass


class PickPlaceExecutor:
    def __init__(self):
        self.target_category = None
        self.cube_category_topic = rospy.get_param(
            '~cube_category_topic', '/cube_category')
        self.category_to_model = rospy.get_param('~category_to_model', {})
        self.parking_areas = rospy.get_param('~parking_areas', {})
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
        self.grasp_prepare_retries = int(
            rospy.get_param('~grasp_prepare_retries', 2))
        self.grasp_retry_max_target_shift = float(
            rospy.get_param('~grasp_retry_max_target_shift', 0.12))
        self.failure_search_retry_delay = float(
            rospy.get_param('~failure_search_retry_delay', 1.0))
        self.failure_search_offsets = rospy.get_param(
            '~failure_search_offsets', [[0.0, 0.0], [0.12, 0.0],
                                        [-0.12, 0.0], [0.0, 0.12],
                                        [0.0, -0.12]])
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
        self.fine_align_distance = float(
            rospy.get_param('~fine_align_distance', 0.08))
        self.fine_align_timeout = float(
            rospy.get_param('~fine_align_timeout', 8.0))
        self.fine_align_rate = float(
            rospy.get_param('~fine_align_rate', 20.0))
        self.fine_align_max_linear_speed = float(
            rospy.get_param('~fine_align_max_linear_speed', 0.04))
        self.fine_align_max_angular_speed = float(
            rospy.get_param('~fine_align_max_angular_speed', 0.30))
        self.fine_align_linear_gain = float(
            rospy.get_param('~fine_align_linear_gain', 0.8))
        self.fine_align_angular_gain = float(
            rospy.get_param('~fine_align_angular_gain', 0.8))
        self.fine_align_yaw_tolerance = float(
            rospy.get_param('~fine_align_yaw_tolerance', 0.04))
        self.align_z_tolerance = float(
            rospy.get_param('~align_z_tolerance', 0.03))
        self.coarse_search_pose = rospy.get_param(
            '~coarse_search_pose', {})
        self.area_search_waypoints = rospy.get_param(
            '~area_search_waypoints', [])
        self.area_search_timeout = float(
            rospy.get_param('~area_search_timeout', 6.0))
        self.area_search_pan_offsets = [
            float(value) for value in rospy.get_param(
                '~area_search_pan_offsets', [0.25, -0.25])]
        default_view_stages = [
            {'pose': 'observe', 'pan_offsets': []},
            {'pose': 'area_observe',
             'pan_offsets': list(self.area_search_pan_offsets)},
        ]
        self.area_search_view_stages = rospy.get_param(
            '~area_search_view_stages', default_view_stages)
        self.area_search_pan_timeout = float(
            rospy.get_param('~area_search_pan_timeout', 4.0))
        self.area_search_view_settle_time = float(
            rospy.get_param('~area_search_view_settle_time', 0.8))
        self.area_search_standoff_distance = float(
            rospy.get_param('~area_search_standoff_distance', 0.18))
        self.area_search_reorient_yaw_threshold = float(
            rospy.get_param('~area_search_reorient_yaw_threshold', 2.50))
        self.area_search_passes = int(
            rospy.get_param('~area_search_passes', 2))
        self.fast_area_search = bool(
            rospy.get_param('~fast_area_search', False))
        self.fast_area_stop_on_non_target = bool(
            rospy.get_param('~fast_area_stop_on_non_target', True))
        self.search_target_confirmations = int(
            rospy.get_param('~search_target_confirmations', 2))
        self.search_target_confirm_timeout = float(
            rospy.get_param('~search_target_confirm_timeout', 4.0))
        self.search_target_confirm_distance = float(
            rospy.get_param('~search_target_confirm_distance', 0.05))
        self.search_areas = rospy.get_param('~search_areas', {})
        self.vision_reset_service = rospy.get_param(
            '~vision_reset_service', '/cube_vision/reset')
        self.align_fresh_pose_samples = int(
            rospy.get_param('~align_fresh_pose_samples', 3))
        self.align_fresh_pose_timeout = float(
            rospy.get_param('~align_fresh_pose_timeout', 3.0))
        self.align_fresh_pose_max_span = float(
            rospy.get_param('~align_fresh_pose_max_span', 0.02))
        self.align_settle_time = float(
            rospy.get_param('~align_settle_time', 0.5))
        self.max_grasp_offset = float(
            rospy.get_param('~max_grasp_offset', 0.015))
        self.search_rotation_speed = float(
            rospy.get_param('~search_rotation_speed', 0.20))
        self.search_rotation_angle = float(
            rospy.get_param('~search_rotation_angle', 2.0 * math.pi))
        self.search_rotation_timeout = float(
            rospy.get_param('~search_rotation_timeout', 40.0))
        self.search_rotation_rate = float(
            rospy.get_param('~search_rotation_rate', 20.0))
        self.search_rotation_max_drift = float(
            rospy.get_param('~search_rotation_max_drift', 0.08))
        self.search_settle_time = float(
            rospy.get_param('~search_settle_time', 0.5))
        self.coarse_center_camera_frame = rospy.get_param(
            '~coarse_center_camera_frame', 'camera_depth_optical_frame')
        self.coarse_center_enabled = bool(
            rospy.get_param('~coarse_center_enabled', True))
        self.coarse_center_tolerance = float(
            rospy.get_param('~coarse_center_tolerance', 0.08))
        self.coarse_center_max_angle = float(
            rospy.get_param('~coarse_center_max_angle', 0.65))
        self.coarse_center_timeout = float(
            rospy.get_param('~coarse_center_timeout', 2.5))
        self.coarse_center_gain = float(
            rospy.get_param('~coarse_center_gain', 1.5))
        self.coarse_center_max_speed = float(
            rospy.get_param('~coarse_center_max_speed', 0.30))
        self.search_direct_max_distance = float(
            rospy.get_param('~search_direct_max_distance', 0.30))
        self.search_direct_timeout = float(
            rospy.get_param('~search_direct_timeout', 10.0))
        self.search_direct_max_linear_speed = float(
            rospy.get_param('~search_direct_max_linear_speed', 0.15))
        self.search_direct_max_angular_speed = float(
            rospy.get_param('~search_direct_max_angular_speed', 0.50))
        self.parking_footprint = rospy.get_param('~parking_footprint', [])
        self.parking_footprint_margin = float(
            rospy.get_param('~parking_footprint_margin', 0.01))
        self.parking_position_tolerance = float(
            rospy.get_param('~parking_position_tolerance', 0.015))
        self.parking_yaw_tolerance = float(
            rospy.get_param('~parking_yaw_tolerance', 0.04))
        self.parking_timeout = float(
            rospy.get_param('~parking_timeout', 10.0))
        self.navigation_ready_timeout = float(
            rospy.get_param('~navigation_ready_timeout', 60.0))
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
        self.vision_pose_category = 'unknown'
        self.vision_pose_confidence = 0.0
        self.vision_pose_received = rospy.Time(0)
        self.vision_pose_seq = 0
        self.vision_reset = None
        self.seen_non_target_cubes = []
        self.search_detection_dedupe_distance = float(
            rospy.get_param('~search_detection_dedupe_distance', 0.05))
        self.ready = False
        self.grasp_state = 'IDLE'
        self.attached_model = ''
        self.attach_offset = None
        self.attach_offset_history = []
        self.grasp_tcp_in_base = None
        self.transport_tcp_in_base = None
        self.place_tcp_in_base = None
        self.target_was_located = False
        self.grasp_completed = False
        self.startup_prepared = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.nav_client = actionlib.SimpleActionClient(
            '/move_base', MoveBaseAction)
        self.arm_client = actionlib.SimpleActionClient(
            '/arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        self.clear_costmaps = rospy.ServiceProxy(
            '/move_base/clear_costmaps', Empty)

        self.gripper_pub = rospy.Publisher(
            '/gripper_controller/command', Float64, queue_size=1)
        self.state_pub = rospy.Publisher(
            '~state', String, queue_size=1, latch=True)
        self.result_pub = rospy.Publisher(
            '~result', String, queue_size=1, latch=True)
        self.detected_category_pub = rospy.Publisher(
            '~detected_category', String, queue_size=1, latch=True)
        self.detected_area_pub = rospy.Publisher(
            '~detected_area', String, queue_size=1, latch=True)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.vision_reset = rospy.ServiceProxy(
            self.vision_reset_service, Empty)

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
        rospy.Subscriber(
            self.cube_category_topic, String, self._category_command_cb,
            queue_size=1)
        rospy.Subscriber('/cube_vision/confidence', Float32, self._confidence_cb)
        rospy.Subscriber('/cube_vision/pose', PoseStamped, self._vision_pose_cb)
        rospy.Subscriber('/grasp_attach/ready', Bool, self._ready_cb)
        rospy.Subscriber('/grasp_attach/state', String, self._grasp_state_cb)
        rospy.Subscriber(
            '/grasp_attach/attached_model', String, self._attached_model_cb)
        rospy.Subscriber('/grasp_attach/offset', Point, self._offset_cb)

        rospy.on_shutdown(self._cancel_actions)
        self._validate_config()
        rospy.set_param('/gazebo_success', 0)

    def _validate_config(self):
        for category in TARGET_CATEGORIES:
            if category not in self.category_to_model:
                raise rospy.ROSInitException(
                    'category has no model mapping: ' + category)
            if category not in self.parking_areas:
                raise rospy.ROSInitException(
                    'category has no parking area: ' + category)
        required_pose = ('x', 'y', 'yaw')
        if (not isinstance(self.coarse_search_pose, dict)
                or any(key not in self.coarse_search_pose
                       for key in required_pose)):
            raise rospy.ROSInitException('coarse_search_pose is incomplete')
        if not all(math.isfinite(float(self.coarse_search_pose[key]))
                   for key in required_pose):
            raise rospy.ROSInitException(
                'coarse_search_pose contains invalid values')
        if not self._inside_map(
                float(self.coarse_search_pose['x']),
                float(self.coarse_search_pose['y'])):
            raise rospy.ROSInitException(
                'coarse_search_pose is outside map bounds')
        if (not isinstance(self.area_search_waypoints, list)
                or not self.area_search_waypoints):
            raise rospy.ROSInitException('area_search_waypoints is empty')
        configured_areas = []
        for index, waypoint in enumerate(self.area_search_waypoints):
            if (not isinstance(waypoint, dict)
                    or any(key not in waypoint
                           for key in ('area', 'x', 'y', 'yaw'))):
                raise rospy.ROSInitException(
                    'area_search_waypoints[{}] is incomplete'.format(index))
            if not all(math.isfinite(float(waypoint[key]))
                        for key in ('x', 'y', 'yaw')):
                raise rospy.ROSInitException(
                    'area_search_waypoints[{}] contains invalid values'.format(
                        index))
            if not self._inside_map(
                    float(waypoint['x']), float(waypoint['y'])):
                raise rospy.ROSInitException(
                    'area_search_waypoints[{}] is outside map bounds'.format(
                        index))
            configured_areas.append(str(waypoint['area']))
        if configured_areas != ['area_a', 'area_b', 'area_c']:
            raise rospy.ROSInitException(
                'area_search_waypoints must be ordered area_a, area_b, area_c')
        if (not math.isfinite(self.area_search_timeout)
                or self.area_search_timeout <= 0.0):
            raise rospy.ROSInitException('area_search_timeout must be positive')
        if (not self.area_search_pan_offsets
                or not all(math.isfinite(value) and value != 0.0
                           for value in self.area_search_pan_offsets)):
            raise rospy.ROSInitException(
                'area_search_pan_offsets must contain finite nonzero values')
        if (not math.isfinite(self.area_search_pan_timeout)
                or self.area_search_pan_timeout <= 0.0):
            raise rospy.ROSInitException(
                'area_search_pan_timeout must be positive')
        if (not math.isfinite(self.area_search_view_settle_time)
                or self.area_search_view_settle_time <= 0.0):
            raise rospy.ROSInitException(
                'area_search_view_settle_time must be positive')
        if (not isinstance(self.area_search_view_stages, list)
                or not self.area_search_view_stages):
            raise rospy.ROSInitException(
                'area_search_view_stages must be a non-empty list')
        for index, stage in enumerate(self.area_search_view_stages):
            if not isinstance(stage, dict) or 'pose' not in stage:
                raise rospy.ROSInitException(
                    'area_search_view_stages[{}] is incomplete'.format(index))
            pose_name = str(stage['pose'])
            if pose_name not in self.arm_poses:
                raise rospy.ROSInitException(
                    'area_search_view_stages[{}] references missing pose {}'.format(
                        index, pose_name))
            offsets = stage.get('pan_offsets', [])
            if (not isinstance(offsets, list)
                    or not all(math.isfinite(float(value)) for value in offsets)):
                raise rospy.ROSInitException(
                    'area_search_view_stages[{}] pan_offsets are invalid'.format(
                        index))
        if (not math.isfinite(self.area_search_standoff_distance)
                or self.area_search_standoff_distance <= 0.0
                or self.area_search_standoff_distance > 0.30):
            raise rospy.ROSInitException(
                'area_search_standoff_distance must be in (0, 0.30]')
        if (not math.isfinite(self.area_search_reorient_yaw_threshold)
                or self.area_search_reorient_yaw_threshold <= 0.0
                or self.area_search_reorient_yaw_threshold > math.pi):
            raise rospy.ROSInitException(
                'area_search_reorient_yaw_threshold must be in (0, pi]')
        if self.area_search_passes < 1:
            raise rospy.ROSInitException(
                'area_search_passes must be at least one')
        if self.search_target_confirmations < 1:
            raise rospy.ROSInitException(
                'search_target_confirmations must be at least one')
        for name, value in (
                ('coarse_center_tolerance', self.coarse_center_tolerance),
                ('coarse_center_max_angle', self.coarse_center_max_angle),
                ('coarse_center_timeout', self.coarse_center_timeout),
                ('coarse_center_gain', self.coarse_center_gain),
                ('coarse_center_max_speed', self.coarse_center_max_speed)):
            if not math.isfinite(value) or value <= 0.0:
                raise rospy.ROSInitException(name + ' must be positive')
        if self.coarse_center_max_angle > math.pi:
            raise rospy.ROSInitException(
                'coarse_center_max_angle must not exceed pi')
        for name, value in (
                ('search_target_confirm_timeout',
                 self.search_target_confirm_timeout),
                ('search_target_confirm_distance',
                 self.search_target_confirm_distance)):
            if not math.isfinite(value) or value <= 0.0:
                raise rospy.ROSInitException(name + ' must be positive')
        if (not math.isfinite(self.search_rotation_speed)
                or self.search_rotation_speed == 0.0):
            raise rospy.ROSInitException(
                'search_rotation_speed must be finite and nonzero')
        for name, value in (
                ('search_rotation_angle', self.search_rotation_angle),
                ('search_rotation_timeout', self.search_rotation_timeout),
                ('search_rotation_rate', self.search_rotation_rate),
                ('search_rotation_max_drift', self.search_rotation_max_drift)):
            if not math.isfinite(value) or value <= 0.0:
                raise rospy.ROSInitException(name + ' must be positive')
        if (not math.isfinite(self.search_detection_dedupe_distance)
                or self.search_detection_dedupe_distance <= 0.0):
            raise rospy.ROSInitException(
                'search_detection_dedupe_distance must be positive')
        if self.search_rotation_angle > 2.0 * math.pi + 0.01:
            raise rospy.ROSInitException(
                'search_rotation_angle must not exceed one revolution')
        if self.grasp_prepare_retries < 0:
            raise rospy.ROSInitException(
                'grasp_prepare_retries must not be negative')
        if (not math.isfinite(self.grasp_retry_max_target_shift)
                or self.grasp_retry_max_target_shift <= 0.0):
            raise rospy.ROSInitException(
                'grasp_retry_max_target_shift must be positive')
        if (not math.isfinite(self.failure_search_retry_delay)
                or self.failure_search_retry_delay <= 0.0):
            raise rospy.ROSInitException(
                'failure_search_retry_delay must be positive')
        if (not isinstance(self.failure_search_offsets, list)
                or not self.failure_search_offsets):
            raise rospy.ROSInitException(
                'failure_search_offsets must be a non-empty list')
        for offset in self.failure_search_offsets:
            if (not isinstance(offset, list) or len(offset) != 2
                    or not all(math.isfinite(float(value)) for value in offset)
                    or math.hypot(float(offset[0]), float(offset[1])) > 0.25):
                raise rospy.ROSInitException(
                    'failure_search_offsets must contain finite offsets <= 0.25 m')
        for name, value in (
                ('search_direct_max_distance',
                 self.search_direct_max_distance),
                ('search_direct_timeout', self.search_direct_timeout),
                ('search_direct_max_linear_speed',
                 self.search_direct_max_linear_speed),
                ('search_direct_max_angular_speed',
                 self.search_direct_max_angular_speed)):
            if not math.isfinite(value) or value <= 0.0:
                raise rospy.ROSInitException(name + ' must be positive')
        for name, value in (
                ('fine_align_distance', self.fine_align_distance),
                ('fine_align_timeout', self.fine_align_timeout),
                ('fine_align_rate', self.fine_align_rate),
                ('fine_align_max_linear_speed',
                 self.fine_align_max_linear_speed),
                ('fine_align_max_angular_speed',
                 self.fine_align_max_angular_speed),
                ('fine_align_linear_gain', self.fine_align_linear_gain),
                ('fine_align_angular_gain', self.fine_align_angular_gain),
                ('fine_align_yaw_tolerance', self.fine_align_yaw_tolerance)):
            if not math.isfinite(value) or value <= 0.0:
                raise rospy.ROSInitException(name + ' must be positive')
        required_arm_poses = [
            'navigation', 'observe', 'area_observe', 'grasp_approach',
            'grasp', 'grasp_lift', 'transport', 'place']
        required_arm_poses.extend(
            str(stage['pose']) for stage in self.area_search_view_stages)
        for name in dict.fromkeys(required_arm_poses):
            self._read_arm_pose(name)
        for name, value in (
                ('gripper_open', self.gripper_open),
                ('gripper_close', self.gripper_close)):
            if not math.isfinite(value) or not -1.51 <= value <= 1.51:
                raise rospy.ROSInitException(name + ' is outside r_joint limits')
        required_bounds = ('x_min', 'x_max', 'y_min', 'y_max')
        if any(key not in self.map_bounds for key in required_bounds):
            raise rospy.ROSInitException('map_bounds is incomplete')
        if (not isinstance(self.parking_footprint, list)
                or len(self.parking_footprint) < 3):
            raise rospy.ROSInitException('parking_footprint is incomplete')
        for vertex in self.parking_footprint:
            if (not isinstance(vertex, list) or len(vertex) != 2
                    or not all(math.isfinite(float(value)) for value in vertex)):
                raise rospy.ROSInitException('parking_footprint contains invalid values')
        for name, value in (
                ('parking_footprint_margin', self.parking_footprint_margin),
                ('parking_position_tolerance', self.parking_position_tolerance),
                ('parking_yaw_tolerance', self.parking_yaw_tolerance),
                ('parking_timeout', self.parking_timeout)):
            if not math.isfinite(value) or value <= 0.0:
                raise rospy.ROSInitException(name + ' must be positive')

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

    def _category_command_cb(self, msg):
        category = msg.data.strip().lower()
        if category == 'electronic':
            category = 'electronics'
        if category not in TARGET_CATEGORIES:
            rospy.logwarn(
                'ignoring invalid cube_category=%r; expected one of %s',
                msg.data, ', '.join(TARGET_CATEGORIES))
            return
        if self.target_category is not None:
            rospy.logwarn(
                'ignoring cube_category=%s; task already selected %s',
                category, self.target_category)
            return
        self.target_category = category
        rospy.loginfo('accepted cube_category=%s', category)

    def _confidence_cb(self, msg):
        self.confidence = float(msg.data)

    def _vision_pose_cb(self, msg):
        self.vision_pose = msg
        # cube_vision publishes category/confidence immediately before pose;
        # retain that association so stale metadata cannot label a new pose.
        self.vision_pose_category = self.category
        self.vision_pose_confidence = self.confidence
        self.vision_pose_received = rospy.Time.now()
        self.vision_pose_seq += 1

    def _ready_cb(self, msg):
        self.ready = bool(msg.data)

    def _grasp_state_cb(self, msg):
        self.grasp_state = msg.data

    def _attached_model_cb(self, msg):
        self.attached_model = msg.data

    def _offset_cb(self, msg):
        self.attach_offset = (float(msg.x), float(msg.y), float(msg.z))

    def _set_state(self, state):
        rospy.loginfo('pick_place state: %s', state)
        self.state_pub.publish(String(data=state))

    @staticmethod
    def _log_timing(stage, started, detail=''):
        rospy.loginfo(
            'timing %s: %.3f s%s', stage, time.monotonic() - started,
            ' ' + detail if detail else '')

    def wait_for_target_category(self):
        self._prepare_task_start()
        self._set_state('WAITING_FOR_CATEGORY')
        rospy.loginfo(
            'waiting for std_msgs/String on %s: %s',
            self.cube_category_topic, ', '.join(TARGET_CATEGORIES))
        while not rospy.is_shutdown() and self.target_category is None:
            time.sleep(0.05)
        if self.target_category is None:
            raise RuntimeError('category wait interrupted by shutdown')

    def _prepare_task_start(self):
        if self.startup_prepared:
            return
        started = time.monotonic()
        self._set_state('PREPARE_WHILE_WAITING')
        if not self.arm_poses_verified:
            raise RuntimeError(
                'arm_poses_verified is false; confirm configured arm poses are safe')
        if not self._wait_for_action_server(self.nav_client, 'move_base'):
            raise RuntimeError('move_base action server unavailable')
        if not self._wait_for_action_server(
                self.arm_client, 'arm trajectory'):
            raise RuntimeError('arm trajectory action server unavailable')
        if not self._wait_joint_state(8.0):
            raise RuntimeError('joint state unavailable')

        self._command_gripper(self.gripper_open, 'open')
        if self.grasp_state != 'IDLE' or self.attached_model:
            deadline = rospy.Time.now() + rospy.Duration(2.0)
            while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                if self.grasp_state == 'IDLE' and not self.attached_model:
                    break
                rospy.sleep(0.05)
        if self.grasp_state != 'IDLE' or self.attached_model:
            raise RuntimeError('grasp backend is not idle')

        # Only prepare actuators and localization here. The base remains at
        # its spawn pose until a valid category starts the search.
        self._move_arm('navigation')
        self._wait_for_navigation_ready()
        self.startup_prepared = True
        self._log_timing('startup.prepare_while_waiting', started)

    def _stop_base(self):
        self.cmd_vel_pub.publish(Twist())

    def _cancel_actions(self):
        self._stop_base()
        self.nav_client.cancel_all_goals()
        self.arm_client.cancel_all_goals()

    def _recover_after_failure(self):
        """Leave the robot in a safe, reusable state after a task error."""
        self._set_state('RECOVER_FAILURE')
        self._cancel_actions()

        try:
            self._command_gripper(self.gripper_open, 'failure open')
        except (rospy.ROSException, RuntimeError) as exc:
            rospy.logwarn('failure recovery could not open gripper: %s', exc)

        # Give grasp_attach a wall-time window to publish IDLE after opening.
        deadline = time.monotonic() + 2.0
        while (not rospy.is_shutdown() and time.monotonic() < deadline
               and (self.grasp_state != 'IDLE' or self.attached_model)):
            time.sleep(0.05)

        if self.grasp_state != 'IDLE' or self.attached_model:
            rospy.logwarn(
                'failure recovery will not move arm while attachment remains: '
                'state=%s model=%s', self.grasp_state, self.attached_model)
            self._stop_base()
            return False

        try:
            self._move_arm('navigation')
        except (rospy.ROSException, RuntimeError) as exc:
            rospy.logwarn('failure recovery could not raise arm: %s', exc)
            self._stop_base()
            return False
        self._stop_base()
        return True

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

    def _move_arm(self, name, joint1_offset=0.0):
        positions, duration = self._read_arm_pose(name)
        if joint1_offset:
            positions[0] += float(joint1_offset)
            if not -3.14 <= positions[0] <= 3.14:
                raise RuntimeError(
                    '{} joint1 compensation is outside limits'.format(name))
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
        current = [self.joint_values.get(joint) for joint in JOINT_NAMES]
        diagnostics = []
        for joint, target, actual in zip(JOINT_NAMES, positions, current):
            if actual is None:
                diagnostics.append('{}=unavailable'.format(joint))
            else:
                diagnostics.append(
                    '{} target={:.3f} actual={:.3f} error={:.3f}'.format(
                        joint, target, actual, actual - target))
        raise RuntimeError(
            '{} arm pose did not settle: {}'.format(
                name, ', '.join(diagnostics)))

    def _command_gripper(self, target, label, attachment_satisfies=False):
        self.gripper_pub.publish(Float64(data=target))
        deadline = rospy.Time.now() + rospy.Duration(4.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if attachment_satisfies and self.grasp_state == 'GRASPING':
                return
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

    def _current_known_detection(self, since_received=None):
        pose = self.vision_pose
        received = self.vision_pose_received
        category = self.vision_pose_category
        if pose is None or category not in self.category_to_model:
            return None
        stamp = pose.header.stamp
        age = ((rospy.Time.now() - stamp).to_sec()
               if stamp != rospy.Time(0) else 0.0)
        received_after = since_received is None or received > since_received
        stamp_is_usable = stamp == rospy.Time(0) or age >= 0.0
        if (received_after and pose.header.frame_id and stamp_is_usable
                and age <= self.max_pose_age
                and self.vision_pose_confidence >= self.min_confidence):
            return category, pose
        return None

    def _current_detection(self, since_received=None):
        detection = self._current_known_detection(since_received)
        if detection is not None and detection[0] == self.target_category:
            return detection[1]
        return None

    def _search_area_name(self, pose):
        point = self._point_in_frame(pose, 'map').point
        matches = []
        margin = float(rospy.get_param('~search_area_match_margin', 0.03))
        for name, bounds in self.search_areas.items():
            if all(key in bounds for key in ('x_min', 'x_max', 'y_min', 'y_max')):
                if (float(bounds['x_min']) - margin <= point.x <= float(bounds['x_max']) + margin
                        and float(bounds['y_min']) - margin <= point.y <= float(bounds['y_max']) + margin):
                    matches.append(name)
        return matches[0] if len(matches) == 1 else None

    def _area_observation_yaw(self, area, waypoint):
        """Face the configured source-area center from a search waypoint."""
        configured_yaw = float(waypoint['yaw'])
        bounds = self.search_areas.get(area, {})
        try:
            center_x = 0.5 * (
                float(bounds['x_min']) + float(bounds['x_max']))
            center_y = 0.5 * (
                float(bounds['y_min']) + float(bounds['y_max']))
        except (KeyError, TypeError, ValueError):
            rospy.logwarn(
                'area %s has no complete bounds; using configured yaw %.3f',
                area, configured_yaw)
            return configured_yaw

        dx = center_x - float(waypoint['x'])
        dy = center_y - float(waypoint['y'])
        if math.hypot(dx, dy) < 1e-6:
            rospy.logwarn(
                'area %s waypoint is at its center; using configured yaw %.3f',
                area, configured_yaw)
            return configured_yaw
        yaw = math.atan2(dy, dx)
        yaw_delta = self._wrap_angle(yaw - configured_yaw)
        if abs(yaw_delta) > 0.15:
            rospy.loginfo(
                'area %s observation yaw derived from waypoint to area center: '
                'configured=%.3f derived=%.3f delta=%.3f',
                area, configured_yaw, yaw, yaw_delta)
        return yaw

    def _area_view_joint1_offset(
            self, nominal_joint1, target_yaw, actual_yaw):
        """Return the offset that makes the camera face the area in map yaw."""
        desired_joint1 = self._wrap_angle(target_yaw - actual_yaw)
        candidates = [desired_joint1 + 2.0 * math.pi * turns
                      for turns in (-1, 0, 1)]
        valid_targets = [value for value in candidates
                         if -3.14 <= value <= 3.14]
        if not valid_targets:
            return None
        target_joint1 = min(
            valid_targets, key=lambda value: abs(value - nominal_joint1))
        return target_joint1 - nominal_joint1

    def _classify_search_area(self, pose):
        point = self._point_in_frame(pose, 'map').point
        area = self._search_area_name(pose)
        if area is None:
            raise RuntimeError(
                'vision target is not uniquely inside a search area: map=(%.3f, %.3f)' %
                (point.x, point.y))
        self.detected_area_pub.publish(String(data=area))
        rospy.loginfo('vision target area=%s map=(%.3f, %.3f)', area, point.x, point.y)
        return area

    def _fresh_target_pose(self, barrier_seq, timeout=None):
        timeout = self.align_fresh_pose_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        samples = []
        accepted_seq = barrier_seq
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.vision_pose_seq > accepted_seq:
                detection = self._current_detection()
                accepted_seq = self.vision_pose_seq
                if detection is not None:
                    point = self._point_in_frame(detection, 'map').point
                    samples.append((point.x, point.y, point.z, detection))
                    if len(samples) >= self.align_fresh_pose_samples:
                        values = np.asarray([
                            [s[0], s[1], s[2]]
                            for s in samples[-self.align_fresh_pose_samples:]])
                        if np.max(np.ptp(values, axis=0)) <= self.align_fresh_pose_max_span:
                            return samples[-1][3]
            rospy.sleep(0.05)
        return None

    def _reset_vision_and_observe(self):
        self._stop_base()
        time.sleep(self.align_settle_time)
        self._move_arm('observe')
        barrier = self.vision_pose_seq
        try:
            self.vision_reset()
        except rospy.ServiceException as exc:
            rospy.logwarn('vision reset failed: %s', exc)
        return self._fresh_target_pose(barrier)

    def _center_coarse_view(self, candidate_pose, label):
        """Rotate only enough to bring a rotating-search candidate to center."""
        if not getattr(self, 'coarse_center_enabled', True):
            return False
        camera_frame = getattr(
            self, 'coarse_center_camera_frame', 'camera_depth_optical_frame')
        try:
            camera_point = self._point_in_current_frame(
                candidate_pose, camera_frame).point
            # Gazebo's depth sensor is pi-rolled relative to the URDF optical
            # frame (the same conversion used by cube_vision), so its viewing
            # direction is -Z in camera_depth_optical_frame.
            initial_angle = math.atan2(
                float(camera_point.x), -float(camera_point.z))
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            rospy.logwarn(
                'coarse centering skipped at %s; camera TF unavailable: %s',
                label, exc)
            return False

        if not math.isfinite(initial_angle) or float(camera_point.z) >= 0.0:
            rospy.logwarn(
                'coarse centering skipped at %s; candidate is behind camera',
                label)
            return False
        if abs(initial_angle) <= self.coarse_center_tolerance:
            rospy.loginfo(
                'coarse target already centered at %s: horizontal_error=%.3f',
                label, initial_angle)
            return True
        if abs(initial_angle) > self.coarse_center_max_angle:
            rospy.logwarn(
                'coarse centering skipped at %s; horizontal_error=%.3f exceeds '
                'limit %.3f', label, initial_angle,
                self.coarse_center_max_angle)
            return False

        rospy.loginfo(
            'coarse target seen near image edge at %s; centering view '
            'horizontal_error=%.3f', label, initial_angle)
        self._set_state('CENTER_COARSE_VIEW')
        deadline = time.monotonic() + self.coarse_center_timeout
        period = 1.0 / max(10.0, float(self.search_rotation_rate))
        command = Twist()
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                try:
                    camera_point = self._point_in_current_frame(
                        candidate_pose, camera_frame).point
                    angle = math.atan2(
                        float(camera_point.x), -float(camera_point.z))
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    rospy.logwarn(
                        'coarse centering lost camera TF at %s: %s', label, exc)
                    return False
                if float(camera_point.z) >= 0.0:
                    return False
                if abs(angle) <= self.coarse_center_tolerance:
                    rospy.loginfo(
                        'coarse view centered at %s: horizontal_error=%.3f',
                        label, angle)
                    return True
                command.linear.x = 0.0
                command.linear.y = 0.0
                command.angular.z = self._clamp(
                    self.coarse_center_gain * angle,
                    self.coarse_center_max_speed)
                self.cmd_vel_pub.publish(command)
                time.sleep(period)
        finally:
            self._stop_base()
        rospy.logwarn('coarse centering timed out at %s', label)
        return False

    def _confirm_stopped_search_target(self, candidate_pose, label):
        """Accept a rotating-search target only from fresh stationary frames."""
        self._stop_base()
        centered = self._center_coarse_view(candidate_pose, label)
        if not centered:
            rospy.loginfo(
                'coarse centering skipped/failed at %s; keeping stationary '
                'confirmation fallback', label)
        self._set_state('CONFIRM_CUBE')
        time.sleep(self.search_settle_time)
        barrier_seq = self.vision_pose_seq
        try:
            self.vision_reset()
        except rospy.ServiceException as exc:
            rospy.logwarn(
                'vision reset failed during stopped confirmation at %s: %s',
                label, exc)
        fresh_pose = self._fresh_target_pose(
            barrier_seq, timeout=self.search_target_confirm_timeout)
        if fresh_pose is None:
            rospy.logwarn(
                'stopped confirmation produced no fresh target pose at %s',
                label)
            return None

        candidate_point = self._point_in_frame(candidate_pose, 'map').point
        fresh_point = self._point_in_frame(fresh_pose, 'map').point
        shift = math.hypot(
            fresh_point.x - candidate_point.x,
            fresh_point.y - candidate_point.y)
        if shift > self.search_target_confirm_distance:
            rospy.logwarn(
                'stopped confirmation rejected shifted target at %s: %.3f m '
                '> %.3f m', label, shift,
                self.search_target_confirm_distance)
            return None

        self.detected_category_pub.publish(
            String(data=self.target_category))
        rospy.loginfo(
            'stopped target confirmed: category=%s confidence=%.3f '
            'shift=%.3f m at %s', self.target_category,
            self.vision_pose_confidence, shift, label)
        return fresh_pose

    def _is_seen_non_target(self, category, pose):
        point = self._point_in_frame(pose, 'map').point
        for seen_category, seen_x, seen_y in self.seen_non_target_cubes:
            if (category == seen_category
                    and math.hypot(point.x - seen_x, point.y - seen_y)
                    <= self.search_detection_dedupe_distance):
                return True
        return False

    def _seen_category_at_pose(self, pose):
        point = self._point_in_frame(pose, 'map').point
        for seen_category, seen_x, seen_y in self.seen_non_target_cubes:
            if math.hypot(point.x - seen_x, point.y - seen_y) <= \
                    self.search_detection_dedupe_distance:
                return seen_category
        return None

    def _detections_share_pose(self, first, second):
        first_point = self._point_in_frame(first[1], 'map').point
        second_point = self._point_in_frame(second[1], 'map').point
        return math.hypot(
            first_point.x - second_point.x,
            first_point.y - second_point.y) <= \
            self.search_detection_dedupe_distance

    def _record_non_target(self, category, pose):
        point = self._point_in_frame(pose, 'map').point
        self.seen_non_target_cubes.append((category, point.x, point.y))
        self.detected_category_pub.publish(String(data=category))
        rospy.loginfo(
            'confirmed non-target cube: category=%s confidence=%.3f '
            'map=(%.3f, %.3f)',
            category, self.vision_pose_confidence, point.x, point.y)

    def _fresh_detection(self, since_received=None, timeout=None):
        timeout = self.detection_timeout if timeout is None else timeout
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            pose = self._current_detection(since_received)
            if pose is not None:
                return pose
            rospy.sleep(0.05)
        return None

    def _fresh_known_detection(self, since_received, timeout):
        deadline = time.monotonic() + timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            detection = self._current_known_detection(since_received)
            if detection is not None:
                return detection
            time.sleep(0.05)
        return None

    def _fresh_search_detection(self, barrier_seq, timeout):
        deadline = time.monotonic() + timeout
        accepted_seq = barrier_seq
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.vision_pose_seq > accepted_seq:
                accepted_seq = self.vision_pose_seq
                detection = self._current_known_detection()
                if detection is not None:
                    return detection
            time.sleep(0.05)
        return None

    def _confirm_search_target(self, initial_detection, label,
                               expected_area=None, center_view=False):
        # A rotating view can leave the candidate at the image edge. Stop
        # before any confirmation reset so all subsequent frames are static.
        self._stop_base()
        if center_view:
            centered = self._center_coarse_view(initial_detection[1], label)
            if not centered:
                rospy.loginfo(
                    'coarse centering skipped/failed at %s; keeping stationary '
                    'confirmation fallback', label)
        detections = [initial_detection]
        initial_point = self._point_in_frame(initial_detection[1], 'map').point
        for confirmation in range(self.search_target_confirmations):
            time.sleep(self.area_search_view_settle_time)
            barrier_seq = self.vision_pose_seq
            try:
                self.vision_reset()
            except rospy.ServiceException as exc:
                rospy.logwarn(
                    'vision reset failed during %s target confirmation: %s',
                    label, exc)
            self._set_state('CONFIRM_TARGET')
            detection = self._fresh_search_detection(
                barrier_seq, self.search_target_confirm_timeout)
            if detection is None:
                rospy.logwarn(
                    '%s target confirmation %d/%d produced no stable pose',
                    label, confirmation + 1,
                    self.search_target_confirmations)
                continue
            if (expected_area is not None
                    and self._search_area_name(detection[1]) != expected_area):
                rospy.logwarn(
                    '%s target confirmation %d/%d produced a pose outside '
                    'expected %s area', label, confirmation + 1,
                    self.search_target_confirmations, expected_area)
                continue
            point = self._point_in_frame(detection[1], 'map').point
            distance = math.hypot(
                point.x - initial_point.x, point.y - initial_point.y)
            if distance > self.search_target_confirm_distance:
                rospy.logwarn(
                    '%s target confirmation %d/%d rejected distant pose: '
                    'category=%s distance=%.3f',
                    label, confirmation + 1,
                    self.search_target_confirmations,
                    detection[0], distance)
                continue
            detections.append(detection)
            rospy.loginfo(
                '%s target confirmation %d/%d sampled: category=%s '
                'distance=%.3f',
                label, confirmation + 1,
                self.search_target_confirmations,
                detection[0], distance)
        required_votes = (self.search_target_confirmations + 2) // 2
        counts = collections.Counter(item[0] for item in detections)
        category, votes = counts.most_common(1)[0]
        if self.target_category in counts and len(counts) > 1:
            rospy.logwarn(
                '%s target classification conflicted across independent '
                'windows: votes=%s; retrying another view',
                label, dict(counts))
            return False, None
        if votes < required_votes:
            rospy.logwarn(
                '%s independent classification inconclusive: votes=%s '
                'required=%d',
                label, dict(counts), required_votes)
            return False, None
        winner = next(
            item for item in reversed(detections) if item[0] == category)
        rospy.loginfo(
            '%s independent classification accepted: category=%s votes=%d/%d',
            label, category, votes,
            self.search_target_confirmations + 1)
        return True, winner

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

    def _point_in_current_frame(self, pose, target_frame):
        """Transform a stationary map pose with the latest live robot TF."""
        point = PointStamped()
        point.header.frame_id = pose.header.frame_id
        point.header.stamp = rospy.Time(0)
        point.point = pose.pose.position
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, point.header.frame_id, rospy.Time(0),
                rospy.Duration(0.5))
            return tf2_geometry_msgs.do_transform_point(point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            raise RuntimeError(
                'live vision pose TF unavailable: {}'.format(exc))

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

    def _base_pose_odom(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'base_footprint', rospy.Time(0), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            raise RuntimeError('odom to base TF unavailable: {}'.format(exc))
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

    def _navigate(self, x, y, yaw, label, retries=None):
        if not self._inside_map(x, y):
            rospy.logwarn('%s goal is outside map: (%.3f, %.3f)', label, x, y)
            return False
        retries = self.nav_retries if retries is None else retries
        for attempt in range(retries + 1):
            self.nav_client.send_goal(self._move_base_goal(x, y, yaw))
            deadline = time.monotonic() + self.nav_timeout
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if (label.startswith('parking approach')
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

    def _direct_search_move(self, x, y, label, hold_yaw=None):
        base_x, base_y, current_yaw = self._base_pose_map()
        if hold_yaw is None:
            hold_yaw = current_yaw
        distance = math.hypot(x - base_x, y - base_y)
        if distance > self.search_direct_max_distance:
            rospy.loginfo(
                '%s is %.3f m away; keeping move_base for this transition',
                label, distance)
            return None

        self.nav_client.cancel_all_goals()
        self._set_state('DIRECT_SEARCH_MOVE')
        deadline = time.monotonic() + self.search_direct_timeout
        period = 1.0 / self.fine_align_rate
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                base_x, base_y, base_yaw = self._base_pose_map()
                error_map_x = x - base_x
                error_map_y = y - base_y
                error_xy = math.hypot(error_map_x, error_map_y)
                yaw_error = self._wrap_angle(hold_yaw - base_yaw)
                if (error_xy <= self.align_xy_tolerance
                        and abs(yaw_error)
                        <= self.fine_align_yaw_tolerance):
                    rospy.loginfo(
                        '%s direct move reached: xy_error=%.3f yaw_error=%.3f',
                        label, error_xy, yaw_error)
                    return True

                cos_yaw = math.cos(base_yaw)
                sin_yaw = math.sin(base_yaw)
                command = Twist()
                command.linear.x = self._clamp(
                    self.fine_align_linear_gain * (
                        cos_yaw * error_map_x + sin_yaw * error_map_y),
                    self.search_direct_max_linear_speed)
                command.linear.y = self._clamp(
                    self.fine_align_linear_gain * (
                        -sin_yaw * error_map_x + cos_yaw * error_map_y),
                    self.search_direct_max_linear_speed)
                command.angular.z = self._clamp(
                    self.fine_align_angular_gain * yaw_error,
                    self.search_direct_max_angular_speed)
                self.cmd_vel_pub.publish(command)
                time.sleep(period)
        finally:
            self._stop_base()

        rospy.logwarn('%s direct move timed out', label)
        return False

    def _rotate_search_at_pose(
            self, waypoint, label='coarse search', navigate_to_pose=True):
        search_x = float(waypoint['x'])
        search_y = float(waypoint['y'])
        search_yaw = float(waypoint['yaw'])
        if navigate_to_pose:
            started = time.monotonic()
            if not self._navigate(
                    search_x, search_y, search_yaw, label, retries=0):
                self._log_timing('search.coarse.navigation', started, 'failed')
                rospy.logwarn(
                    '%s navigation failed; trying fixed area searches', label)
                return None
            self._log_timing('search.coarse.navigation', started)
        self._stop_base()
        started = time.monotonic()
        time.sleep(self.search_settle_time)
        self._log_timing('search.coarse.settle', started)
        started = time.monotonic()
        self._move_arm('observe')
        self._log_timing('search.coarse.observe_arm', started)
        since_received = self.vision_pose_received
        initial_detection = self._current_known_detection(since_received)
        if initial_detection is not None:
            category, pose = initial_detection
            started = time.monotonic()
            if category == self.target_category:
                fresh_pose = self._confirm_stopped_search_target(pose, label)
                confirmed = fresh_pose is not None
                confirmed_detection = (
                    (category, fresh_pose) if confirmed else None)
            else:
                confirmed, confirmed_detection = self._confirm_search_target(
                    initial_detection, label)
            self._log_timing(
                'search.coarse.stationary_confirm', started,
                'category={} accepted={}'.format(category, confirmed))
            if confirmed:
                category, pose = confirmed_detection
                if category == self.target_category:
                    rospy.loginfo(
                        '%s target category=%s confirmed at initial coarse '
                        'view; entering fixed-area search', label, category)
                    return pose
                if not self._is_seen_non_target(category, pose):
                    self._record_non_target(category, pose)
                rospy.loginfo(
                    '%s static view confirmed non-target=%s; '
                    'continuing coarse rotation', label, category)
                since_received = self.vision_pose_received
            else:
                since_received = self.vision_pose_received

        self._set_state('ROTATE_SEARCH')
        rospy.logwarn(
            'rotating with observe arm down via direct /cmd_vel; '
            'move_base collision checking is bypassed')
        start_x, start_y, previous_yaw = self._base_pose_odom()
        accumulated_yaw = 0.0
        rotation_started = time.monotonic()
        deadline = time.monotonic() + self.search_rotation_timeout
        command = Twist()
        command.angular.z = self.search_rotation_speed
        period = 1.0 / self.search_rotation_rate
        try:
            while (not rospy.is_shutdown()
                   and time.monotonic() < deadline
                   and accumulated_yaw < self.search_rotation_angle):
                current_x, current_y, current_yaw = self._base_pose_odom()
                yaw_delta = math.atan2(
                    math.sin(current_yaw - previous_yaw),
                    math.cos(current_yaw - previous_yaw))
                accumulated_yaw += abs(yaw_delta)
                previous_yaw = current_yaw
                drift = math.hypot(current_x - start_x, current_y - start_y)
                if drift > self.search_rotation_max_drift:
                    raise RuntimeError(
                        'search rotation drift exceeded limit: {:.3f} m'.format(
                            drift))

                detection = self._current_known_detection(since_received)
                if detection is not None:
                    category, pose = detection
                    if self._is_seen_non_target(category, pose):
                        rospy.loginfo(
                            '%s skipping previously confirmed non-target=%s '
                            'during coarse rotation', label, category)
                        since_received = self.vision_pose_received
                        _, _, previous_yaw = self._base_pose_odom()
                        self._set_state('ROTATE_SEARCH')
                        continue
                    # Stop for every known cube, not only the requested one.
                    # A moving view can produce a transient category/pose pair;
                    # stationary confirmation must decide before searching on.
                    confirm_started = time.monotonic()
                    confirmed, confirmed_detection = \
                        self._confirm_search_target(
                            detection, label, center_view=True)
                    self._log_timing(
                        'search.coarse.stationary_confirm', confirm_started,
                        'category={} accepted={}'.format(
                            category, confirmed))
                    if not confirmed:
                        since_received = self.vision_pose_received
                        _, _, previous_yaw = self._base_pose_odom()
                        self._set_state('ROTATE_SEARCH')
                        continue

                    category, pose = confirmed_detection
                    if category == self.target_category:
                        self._log_timing(
                            'search.coarse.rotation', rotation_started,
                            'target category={} confirmed at angle={:.2f}; '
                            'entering fixed-area search'.format(
                                category, accumulated_yaw))
                        return pose

                    if not self._is_seen_non_target(category, pose):
                        self._record_non_target(category, pose)
                    self._log_timing(
                        'search.coarse.rotation', rotation_started,
                        'confirmed non-target={} angle={:.2f}; '
                        'continuing coarse rotation'.format(
                            category, accumulated_yaw))
                    since_received = self.vision_pose_received
                    _, _, previous_yaw = self._base_pose_odom()
                    self._set_state('ROTATE_SEARCH')
                    continue

                self.cmd_vel_pub.publish(command)
                time.sleep(period)
        finally:
            self._stop_base()

        self._log_timing(
            'search.coarse.rotation', rotation_started,
            'no_target angle={:.2f}'.format(accumulated_yaw))
        started = time.monotonic()
        self._move_arm('navigation')
        self._log_timing('search.coarse.raise_arm', started)
        if rospy.is_shutdown():
            raise RuntimeError('search rotation interrupted by shutdown')
        rospy.loginfo('%s completed without target; continuing search', label)
        return None

    def _search_near_failure(self, failure_pose):
        """Search around the stopped failure pose without returning home."""
        search_x, search_y, search_yaw = failure_pose
        self._set_state('SEARCH_NEAR_FAILURE')
        rospy.logwarn(
            'retrying target %s around failure pose (%.3f, %.3f, %.3f)',
            self.target_category, search_x, search_y, search_yaw)
        for index, offset in enumerate(self.failure_search_offsets):
            offset_x = float(offset[0])
            offset_y = float(offset[1])
            target_x = search_x + offset_x
            target_y = search_y + offset_y
            if not self._inside_map(target_x, target_y, margin=0.05):
                rospy.logwarn(
                    'failure-local offset %d leaves map; skipping', index)
                continue
            # The observe arm blocks the laser plane. Raise it before each
            # short translation, then lower it only while stopped. Always
            # calculate offsets from the original failure pose so repeated
            # passes cannot drift across the map.
            self._move_arm('navigation')
            moved = self._direct_search_move(
                target_x, target_y,
                'failure-local move {}'.format(index))
            if moved is False:
                rospy.logwarn(
                    'failure-local direct move %d failed; retrying with '
                    'move_base while arm is raised', index)
                moved = self._navigate(
                    target_x, target_y, search_yaw,
                    'failure-local move_base {}'.format(index), retries=0)
            if not moved:
                continue
            waypoint = {'x': target_x, 'y': target_y, 'yaw': search_yaw}
            detection = self._rotate_search_at_pose(
                waypoint, label='failure-local search {}'.format(index),
                navigate_to_pose=False)
            if detection is not None:
                return detection
        self._move_arm('navigation')
        return None

    def _search_area_waypoint(
            self, waypoint, index, _standoff_attempt=False,
            _skip_move=False, _keep_observe=False,
            _stop_on_confirmed_non_target=True, _reuse_observe=False,
            _fast_target_candidate=False, _allow_standoff=True):
        area = str(waypoint['area'])
        label = 'area search {} ({})'.format(index, area)
        target_x = float(waypoint['x'])
        target_y = float(waypoint['y'])
        target_yaw = self._area_observation_yaw(area, waypoint)
        started = time.monotonic()
        move_mode = 'direct'
        if _skip_move:
            moved = True
            self._log_timing(
                'search.{}.move'.format(area), started,
                'mode=existing search pose')
        else:
            moved = self._direct_search_move(
                target_x, target_y, label, hold_yaw=target_yaw)
            if moved is None:
                move_mode = 'move_base'
                moved = self._navigate(
                    target_x, target_y, target_yaw, label, retries=0)
            if not moved:
                self._log_timing(
                    'search.{}.move'.format(area), started,
                    'mode={} failed'.format(move_mode))
                rospy.logwarn('%s move failed; trying next area', label)
                return None
        self._log_timing(
            'search.{}.move'.format(area), started, 'mode=' + move_mode)

        self._stop_base()
        started = time.monotonic()
        time.sleep(self.search_settle_time)
        self._log_timing('search.{}.settle'.format(area), started)
        started = time.monotonic()
        actual_yaw = target_yaw
        if move_mode == 'direct':
            _, _, actual_yaw = self._base_pose_map()
            nominal_joint1 = self._read_arm_pose('area_observe')[0][0]
            yaw_delta = self._wrap_angle(target_yaw - actual_yaw)
            base_joint1_offset = self._area_view_joint1_offset(
                nominal_joint1, target_yaw, actual_yaw)
            if base_joint1_offset is not None:
                if abs(yaw_delta) > self.area_search_reorient_yaw_threshold:
                    rospy.loginfo(
                        '%s base yaw differs by %.3f; using the legal '
                        'absolute arm_joint1 target', label, yaw_delta)
                rospy.loginfo(
                    '%s base aligned; observe joint1 offset=%.3f rad '
                    '(target=%.3f)', label, base_joint1_offset,
                    nominal_joint1 + base_joint1_offset)
            else:
                # A direct known-area transition should normally already face
                # the area. Reorient only when an optional fast path or a
                # controller deviation leaves no legal arm compensation.
                rospy.logwarn(
                    '%s no legal joint1 compensation for yaw delta %.3f; '
                    'reorienting base with move_base', label, yaw_delta)
                self._move_arm('navigation')
                reoriented = self._navigate(
                    target_x, target_y, target_yaw,
                    label + ' yaw reorientation', retries=0)
                if not reoriented:
                    raise RuntimeError(
                        '{} has no usable yaw compensation'.format(label))
                move_mode = 'move_base'
                actual_yaw = target_yaw
                rospy.loginfo(
                    '%s base reoriented only as arm-limit fallback', label)
        detection = None
        non_target_detection = None
        non_target_conflict = False
        view_specs = []
        for stage_index, stage in enumerate(self.area_search_view_stages):
            pose_name = str(stage['pose'])
            view_offsets = [0.0] + [
                float(value) for value in stage.get('pan_offsets', [])]
            nominal_joint1 = self._read_arm_pose(pose_name)[0][0]
            stage_joint1_offset = self._area_view_joint1_offset(
                nominal_joint1, target_yaw, actual_yaw)
            if stage_joint1_offset is None:
                raise RuntimeError(
                    '{} has no {} joint1 compensation within limits'.format(
                        label, pose_name))
            for view_index, pan_offset in enumerate(view_offsets):
                view_specs.append((
                    stage_index, pose_name, view_index, pan_offset,
                    len(view_offsets), nominal_joint1,
                    stage_joint1_offset + pan_offset))

        for (stage_index, pose_name, view_index, pan_offset,
             stage_view_count, nominal_joint1,
             view_joint1_offset) in view_specs:
            if not -3.14 <= nominal_joint1 + view_joint1_offset <= 3.14:
                rospy.logwarn(
                    '%s skipping pan offset %.3f outside joint1 limits',
                    label, pan_offset)
                continue

            started = time.monotonic()
            reuse_view = (
                _reuse_observe and stage_index == 0 and view_index == 0
                and pose_name == str(self.area_search_view_stages[0]['pose'])
                and abs(view_joint1_offset) <= self.joint_tolerance)
            if not reuse_view:
                self._move_arm(
                    pose_name, joint1_offset=view_joint1_offset)
            self._log_timing(
                'search.{}.observe_arm'.format(area), started,
                'stage={} pose={} view={} pan_offset={:.3f} reused={}'.format(
                    stage_index, pose_name, view_index, pan_offset, reuse_view))
            time.sleep(self.area_search_view_settle_time)
            barrier_seq = self.vision_pose_seq
            try:
                self.vision_reset()
            except rospy.ServiceException as exc:
                rospy.logwarn('vision reset failed at %s: %s', label, exc)

            self._set_state('OBSERVE_' + area.upper())
            started = time.monotonic()
            timeout = (self.area_search_timeout
                       if stage_index == 0 and view_index == 0
                       else self.area_search_pan_timeout)
            detection = self._fresh_search_detection(barrier_seq, timeout)
            self._log_timing(
                'search.{}.detection'.format(area), started,
                'view={} pan_offset={:.3f} detected={}'.format(
                    view_index, pan_offset, detection is not None))
            if detection is not None:
                detected_area = self._search_area_name(detection[1])
                if detected_area != area:
                    rospy.logwarn(
                        '%s ignored %s detection outside current %s area',
                        label, detection[0], area)
                    detection = None
                    continue
                if (_fast_target_candidate
                        and detection[0] == self.target_category):
                    rospy.loginfo(
                        '%s fast target candidate accepted for fixed-view '
                        'revalidation', label)
                else:
                    confirmed, confirmed_detection = \
                        self._confirm_search_target(
                            detection, label, expected_area=area,
                            center_view=True)
                    if not confirmed:
                        detection = None
                    else:
                        detection = confirmed_detection
                if (detection is not None
                        and detection[0] != self.target_category
                        and _stop_on_confirmed_non_target):
                    if not self._is_seen_non_target(detection[0], detection[1]):
                        self._record_non_target(detection[0], detection[1])
                    rospy.loginfo(
                        '%s confirmed non-target %s at center view; '
                        'skipping remaining pan views', label, detection[0])
                    if not _keep_observe:
                        self._move_arm('navigation')
                    return None
                if (detection is not None
                        and detection[0] == self.target_category):
                    seen_category = self._seen_category_at_pose(detection[1])
                    if (seen_category is not None
                            and seen_category != self.target_category):
                        rospy.logwarn(
                            '%s rejected category flip at known %s cube pose',
                            label, seen_category)
                        detection = None
                    elif (non_target_detection is not None
                          and non_target_detection[0] != detection[0]
                          and self._detections_share_pose(
                              non_target_detection, detection)):
                        rospy.logwarn(
                            '%s rejected category flip at the same pose: '
                            '%s -> %s', label, non_target_detection[0],
                            detection[0])
                        non_target_conflict = True
                        detection = None
                    else:
                        break
                elif detection is not None:
                    if (non_target_detection is not None
                            and non_target_detection[0] != detection[0]
                            and self._detections_share_pose(
                                non_target_detection, detection)):
                        non_target_conflict = True
                    non_target_detection = detection
                    rospy.loginfo(
                        '%s view %d saw non-target %s; completing remaining '
                        'stationary views for target %s',
                        label, view_index, detection[0], self.target_category)
                    detection = None
            if view_index + 1 < stage_view_count:
                rospy.loginfo(
                    '%s center/corner view missed; trying stationary pan '
                    'offset %.3f rad',
                    label, view_offsets[view_index + 1])

        self._stop_base()
        if detection is None:
            detection = (None if non_target_conflict
                         else non_target_detection)
        if non_target_conflict:
            rospy.logwarn(
                '%s stationary views disagreed on non-target category; '
                'leaving area unclassified', label)
        if detection is None:
            rospy.loginfo(
                '%s completed without a cube after %d stationary views',
                label, len(view_offsets))
            # A cube at the near edge of a source area can be inside the
            # camera near field or outside the arm-camera frustum from the
            # nominal waypoint.  Keep the normal center/corner search fast;
            # only after all stationary views fail, move a bounded distance
            # away from the area's center and repeat the same views once.
            if _allow_standoff and not _standoff_attempt:
                bounds = self.search_areas.get(area, {})
                try:
                    area_center_x = 0.5 * (
                        float(bounds['x_min']) + float(bounds['x_max']))
                    area_center_y = 0.5 * (
                        float(bounds['y_min']) + float(bounds['y_max']))
                except (KeyError, TypeError, ValueError):
                    area_center_x = target_x
                    area_center_y = target_y
                away_x = target_x - area_center_x
                away_y = target_y - area_center_y
                away_norm = math.hypot(away_x, away_y)
                if away_norm < 1e-6:
                    away_x = math.cos(target_yaw)
                    away_y = math.sin(target_yaw)
                    away_norm = 1.0
                standoff_x = target_x + (
                    self.area_search_standoff_distance * away_x / away_norm)
                standoff_y = target_y + (
                    self.area_search_standoff_distance * away_y / away_norm)
                if self._inside_map(standoff_x, standoff_y, margin=0.05):
                    rospy.logwarn(
                        '%s missed from nominal view; retrying %.2f m '
                        'farther from %s center at (%.3f, %.3f)',
                        label, self.area_search_standoff_distance, area,
                        standoff_x, standoff_y)
                    self._move_arm('navigation')
                    standoff_waypoint = dict(waypoint)
                    standoff_waypoint['x'] = standoff_x
                    standoff_waypoint['y'] = standoff_y
                    standoff_pose = self._search_area_waypoint(
                        standoff_waypoint, index,
                        _standoff_attempt=True)
                    if standoff_pose is not None:
                        return standoff_pose
            self._move_arm('navigation')
            return None

        category, pose = detection
        self._set_state('CONFIRM_CUBE')
        if category == self.target_category:
            self.detected_category_pub.publish(String(data=category))
            rospy.loginfo(
                'confirmed target cube: category=%s confidence=%.3f at %s',
                category, self.vision_pose_confidence, area)
            return pose

        if not self._is_seen_non_target(category, pose):
            self._record_non_target(category, pose)
        rospy.loginfo('%s contains non-target %s; trying next area', area, category)
        if not _keep_observe:
            self._move_arm('navigation')
        return None

    def _fast_area_search(self):
        """Search all configured regions from one coarse base pose first.

        The normal fixed-area and full-rotation searches remain below this
        fast path. A center view that independently confirms a non-target is
        sufficient to skip that area's pan views because spawn_cubes places
        exactly one cube in each configured source area.
        """
        self._set_state('FAST_AREA_SEARCH')
        coarse_x = float(self.coarse_search_pose['x'])
        coarse_y = float(self.coarse_search_pose['y'])
        coarse_yaw = float(self.coarse_search_pose['yaw'])
        started = time.monotonic()
        if not self._navigate(
                coarse_x, coarse_y, coarse_yaw,
                'fast area search coarse pose', retries=0):
            rospy.logwarn('fast area search coarse navigation failed')
            return None
        self._stop_base()
        time.sleep(self.search_settle_time)
        self._move_arm(str(self.area_search_view_stages[0]['pose']))

        for index, waypoint in enumerate(self.area_search_waypoints):
            pose = self._search_area_waypoint(
                waypoint, index, _skip_move=True, _keep_observe=True,
                _stop_on_confirmed_non_target=(
                    self.fast_area_stop_on_non_target),
                _reuse_observe=(index == 0),
                _fast_target_candidate=True, _allow_standoff=False)
            if pose is not None:
                rospy.loginfo(
                    'fast area candidate found in %s; revalidating from '
                    'standard fixed observation pose', waypoint['area'])
                self._move_arm('navigation')
                validated_pose = self._search_area_waypoint(
                    waypoint, index)
                if validated_pose is not None:
                    self._log_timing(
                        'search.fast_area.total', started,
                        'source={}'.format(waypoint['area']))
                    return validated_pose
                rospy.logwarn(
                    'fast area candidate in %s failed fixed-view '
                    'revalidation; continuing full search', waypoint['area'])
        self._move_arm('navigation')
        self._log_timing('search.fast_area.total', started, 'missed all areas')
        return None

    def _search(self):
        self._set_state('SEARCH_SOURCE')
        started = time.monotonic()
        if self.fast_area_search:
            pose = self._fast_area_search()
            if pose is not None:
                self._log_timing('search.total', started, 'source=fast_area')
                return pose
        pose = self._rotate_search_at_pose(self.coarse_search_pose)
        if pose is not None:
            area = self._search_area_name(pose)
            if area is not None:
                waypoint_index, waypoint = next(
                    ((index, item) for index, item in enumerate(
                        self.area_search_waypoints)
                     if str(item['area']) == area), (None, None))
                if waypoint is not None:
                    rospy.loginfo(
                        'coarse target confirmed in %s; entering fixed-area '
                        'multi-view search', area)
                    refined = self._search_area_waypoint(
                        waypoint, waypoint_index, _allow_standoff=False)
                    if refined is not None:
                        self._log_timing(
                            'search.total', started,
                            'source=coarse-confirmed-area={}'.format(area))
                        return refined
                    rospy.logwarn(
                        'coarse-confirmed target area %s fixed search missed; '
                        'using the confirmed coarse pose', area)
            self._log_timing('search.total', started, 'source=coarse-confirmed')
            return pose
        for search_pass in range(self.area_search_passes):
            allow_standoff = search_pass > 0
            if not search_pass:
                rospy.loginfo(
                    'coarse search missed; visiting area_a -> area_b -> '
                    'area_c with %d stationary arm views per area before '
                    'any standoff retry',
                    sum(1 + len(stage.get('pan_offsets', []))
                        for stage in self.area_search_view_stages))
            else:
                self._set_state('RETRY_AREA_SEARCH')
                rospy.logwarn(
                    'target missed in nominal A/B/C sweep; starting '
                    'standoff-enabled area search pass %d/%d',
                    search_pass + 1, self.area_search_passes)
            for index, waypoint in enumerate(self.area_search_waypoints):
                pose = self._search_area_waypoint(
                    waypoint, index, _allow_standoff=allow_standoff)
                if pose is not None:
                    self._log_timing(
                        'search.total', started,
                        'source={} pass={}'.format(
                            waypoint['area'], search_pass + 1))
                    return pose
        raise RuntimeError(
            'target was not detected by coarse or repeated multi-view area searches')

    @staticmethod
    def _clamp(value, limit):
        return max(-limit, min(limit, value))

    def _fine_align_to_grasp(self, target_map, rotate=True):
        """Close the final gap with bounded base velocity feedback.

        ``target_map`` is stationary in the map frame.  The target grasp pose
        is recomputed from the live map->base TF on every cycle, so this loop
        corrects odometry/controller error without asking move_base to solve a
        centimeter-scale goal.
        """
        self._set_state('FINE_ALIGN_TO_GRASP')
        deadline = time.monotonic() + self.fine_align_timeout
        period = 1.0 / self.fine_align_rate
        desired_x, desired_y, _ = self.grasp_tcp_in_base
        desired_tcp_angle = math.atan2(desired_y, desired_x)
        command = Twist()
        try:
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                base_x, base_y, base_yaw = self._base_pose_map()
                target_bearing = math.atan2(
                    target_map.y - base_y, target_map.x - base_x)
                grasp_yaw = self._wrap_angle(target_bearing - desired_tcp_angle)
                cos_yaw = math.cos(grasp_yaw)
                sin_yaw = math.sin(grasp_yaw)
                goal_x = target_map.x - (
                    cos_yaw * desired_x - sin_yaw * desired_y)
                goal_y = target_map.y - (
                    sin_yaw * desired_x + cos_yaw * desired_y)
                error_map_x = goal_x - base_x
                error_map_y = goal_y - base_y
                error_xy = math.hypot(error_map_x, error_map_y)
                yaw_error = self._wrap_angle(grasp_yaw - base_yaw)
                rospy.loginfo_throttle(
                    1.0,
                    'fine alignment error: xy=%.3f yaw=%.3f',
                    error_xy, yaw_error)
                if (error_xy <= self.align_xy_tolerance
                        and (not rotate
                             or abs(yaw_error) <= self.fine_align_yaw_tolerance)):
                    return
                if error_xy > self.fine_align_distance:
                    raise RuntimeError(
                        'fine alignment error exceeded limit: {:.3f} m'.format(
                            error_xy))

                # Convert map-frame position error to the base frame expected
                # by /cmd_vel.  Proportional control is capped for safety.
                cos_base = math.cos(base_yaw)
                sin_base = math.sin(base_yaw)
                error_base_x = (cos_base * error_map_x
                                + sin_base * error_map_y)
                error_base_y = (-sin_base * error_map_x
                                + cos_base * error_map_y)
                command.linear.x = self._clamp(
                    self.fine_align_linear_gain * error_base_x,
                    self.fine_align_max_linear_speed)
                command.linear.y = self._clamp(
                    self.fine_align_linear_gain * error_base_y,
                    self.fine_align_max_linear_speed)
                command.angular.z = (self._clamp(
                    self.fine_align_angular_gain * yaw_error,
                    self.fine_align_max_angular_speed) if rotate else 0.0)
                self.cmd_vel_pub.publish(command)
                time.sleep(period)
        finally:
            self._stop_base()
        raise RuntimeError('fine alignment timed out')

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _align_to_grasp_with_fine_tolerance(
            self, first_pose, require_source_area=True):
        self._set_state('ALIGN_TO_GRASP')
        target_pose = first_pose
        if require_source_area:
            self._classify_search_area(target_pose)
        else:
            self.detected_area_pub.publish(String(data='failure_local'))
        first_target = self._point_in_frame(target_pose, 'map').point
        first_base_x, first_base_y, first_base_yaw = self._base_pose_map()
        first_bearing = math.atan2(
            first_target.y - first_base_y, first_target.x - first_base_x)
        observe_yaw_offset = self._wrap_angle(first_bearing - first_base_yaw)
        desired_x, desired_y, _ = self.grasp_tcp_in_base
        desired_tcp_angle = math.atan2(desired_y, desired_x)
        for iteration in range(self.max_align_iterations + 1):
            self._move_arm('navigation')
            target = self._point_in_frame(target_pose, 'map').point
            base_x, base_y, _ = self._base_pose_map()
            target_bearing = math.atan2(target.y - base_y, target.x - base_x)
            grasp_yaw = self._wrap_angle(target_bearing - desired_tcp_angle)
            cos_yaw = math.cos(grasp_yaw)
            sin_yaw = math.sin(grasp_yaw)
            goal_x = target.x - (cos_yaw * desired_x - sin_yaw * desired_y)
            goal_y = target.y - (sin_yaw * desired_x + cos_yaw * desired_y)
            error_x = goal_x - base_x
            error_y = goal_y - base_y
            error_xy = math.hypot(error_x, error_y)
            rospy.loginfo(
                'alignment %d: target=(%.3f, %.3f) goal=(%.3f, %.3f) '
                'error=%.3f grasp_yaw=%.3f',
                iteration, target.x, target.y, goal_x, goal_y,
                error_xy, grasp_yaw)
            if error_xy <= self.fine_align_distance:
                # Finish position and yaw together.  A second observation here
                # would load the nominal observe pose (joint1=+90 degrees)
                # after the base had already moved, causing a visible arm
                # snap and invalidating the fixed-area camera heading.
                self._move_arm('navigation')
                rospy.loginfo(
                    'final grasp alignment: simultaneous XY/yaw correction '
                    'from stable visual target')
                self._fine_align_to_grasp(target, rotate=True)
                return
            if iteration >= self.max_align_iterations:
                break
            scale = min(1.0, self.max_align_step / error_xy)
            step_x = base_x + scale * error_x
            step_y = base_y + scale * error_y
            step_bearing = math.atan2(target.y - step_y, target.x - step_x)
            observe_yaw = self._wrap_angle(step_bearing - observe_yaw_offset)
            if not self._navigate(
                    step_x, step_y, observe_yaw,
                    'visual alignment {}'.format(iteration), retries=0):
                raise RuntimeError('visual alignment navigation failed')
            self._stop_base()
            # Keep the arm in its laser-safe navigation posture during this
            # bounded correction.  Reloading nominal observe here would reset
            # arm_joint1 to +90 degrees after every base move; the stable map
            # target is sufficient for the remaining <= one metre correction.
            rospy.loginfo(
                'visual alignment %d reached intermediate pose; retaining '
                'stable target without observe-arm reload', iteration)
        raise RuntimeError('visual alignment did not converge')

    def _align_to_grasp(self, first_pose, require_source_area=True):
        self._align_to_grasp_with_fine_tolerance(
            first_pose, require_source_area=require_source_area)

    def _prepare_stationary_grasp(self):
        self._set_state('PREPARE_GRASP')
        prepare_started = time.monotonic()
        started = time.monotonic()
        self._move_arm('grasp_approach')
        self._log_timing('grasp.prepare.approach_arm', started)
        started = time.monotonic()
        self._move_arm('grasp')
        self._log_timing('grasp.prepare.descend_arm', started)
        started = time.monotonic()
        deadline = time.monotonic() + 3.0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if (self.ready and self.attach_offset is not None
                    and max(abs(value) for value in self.attach_offset)
                    <= self.max_grasp_offset):
                self._log_timing(
                    'grasp.prepare.ready_offset', started,
                    'ready={} offset={}'.format(
                        self.ready, self.attach_offset))
                self._log_timing('grasp.prepare.total', prepare_started)
                return
            rospy.sleep(0.05)
        raise GraspPreparationError(
            'object is not stably inside the conservative tcp grasp box: '
            'ready={} offset={}'.format(self.ready, self.attach_offset))

    def _grasp_retry_pose_from_offset(self, previous_pose):
        if self.attach_offset is None:
            return None
        if not all(math.isfinite(value) for value in self.attach_offset):
            return None

        offset_point = PointStamped()
        offset_point.header.frame_id = 'tcp_link'
        offset_point.header.stamp = rospy.Time(0)
        offset_point.point.x = self.attach_offset[0]
        offset_point.point.y = self.attach_offset[1]
        offset_point.point.z = self.attach_offset[2]
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'tcp_link', rospy.Time(0), rospy.Duration(0.5))
            target = tf2_geometry_msgs.do_transform_point(
                offset_point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn('grasp retry offset TF unavailable: %s', exc)
            return None

        previous = self._point_in_frame(previous_pose, 'map').point
        shift = math.hypot(
            target.point.x - previous.x, target.point.y - previous.y)
        if shift > self.grasp_retry_max_target_shift:
            rospy.logwarn(
                'rejecting grasp retry offset target %.3f m from the last '
                'visual target', shift)
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time(0)
        pose.pose.position = target.point
        pose.pose.orientation.w = 1.0
        rospy.logwarn(
            'using grasp offset fallback for retry: shift=%.3f m '
            'map=(%.3f, %.3f) offset=%s',
            shift, target.point.x, target.point.y, self.attach_offset)
        return pose

    def _retry_grasp_pose(self, previous_pose):
        # Capture the backend-derived target while the arm is still in the
        # failed grasp posture. Raising the arm changes tcp_link geometry.
        offset_pose = self._grasp_retry_pose_from_offset(previous_pose)
        observed_pose = self._reset_vision_and_observe()
        if observed_pose is not None:
            rospy.loginfo('fresh visual target acquired for grasp retry')
            return observed_pose
        return offset_pose

    def _align_prepare_and_grasp(
            self, first_pose, require_source_area=True):
        target_pose = first_pose
        for attempt in range(self.grasp_prepare_retries + 1):
            if attempt:
                self._set_state('RETRY_GRASP_ALIGN')
                rospy.logwarn(
                    'retrying grasp alignment %d/%d',
                    attempt, self.grasp_prepare_retries)
            self._align_to_grasp(
                target_pose, require_source_area=require_source_area)
            try:
                self._prepare_stationary_grasp()
            except GraspPreparationError as exc:
                if attempt >= self.grasp_prepare_retries:
                    raise
                rospy.logwarn('grasp preparation failed: %s', exc)
                retry_pose = self._retry_grasp_pose(target_pose)
                if retry_pose is None:
                    raise GraspPreparationError(
                        '{}; no fresh retry target available'.format(exc))
                target_pose = retry_pose
                continue
            self._grasp()
            return

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

    def _wait_attached_gripper_settle(self):
        deadline = time.monotonic() + self.settle_timeout
        previous = None
        settled = 0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._check_attachment()
            position = self.joint_values.get('r_joint')
            if position is None:
                time.sleep(0.05)
                continue
            if (previous is not None
                    and abs(position - previous)
                    <= self.settle_position_delta):
                settled += 1
                if settled >= self.settle_samples:
                    return
            else:
                settled = 0
            previous = position
            time.sleep(0.05)
        raise RuntimeError('attached gripper did not settle before transport')

    def _refresh_costmaps_after_pickup(self):
        """Remove the picked cube's stale laser obstacle before transport."""
        global_updates = self._global_costmap_updates
        local_updates = self._local_costmap_updates
        try:
            self.clear_costmaps()
        except rospy.ServiceException as exc:
            raise RuntimeError(
                'costmap reset after pickup failed: {}'.format(exc))

        deadline = time.monotonic() + self.navigation_data_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._check_attachment()
            if (self._global_costmap_updates > global_updates
                    and self._local_costmap_updates > local_updates):
                rospy.loginfo(
                    'costmaps refreshed after pickup: global_updates=%d '
                    'local_updates=%d',
                    self._global_costmap_updates - global_updates,
                    self._local_costmap_updates - local_updates)
                return
            time.sleep(0.05)
        rospy.logwarn(
            'costmap reset completed after pickup, but fresh published '
            'global/local costmaps were not both observed before timeout')

    def _grasp(self):
        self._set_state('GRASP')
        grasp_started = time.monotonic()
        # _prepare_stationary_grasp has already positioned the arm while the
        # base is stopped; only the gripper moves in this state.
        if not self.ready:
            raise RuntimeError('object is not inside the tcp grasp box')
        started = time.monotonic()
        self._command_gripper(
            self.gripper_close, 'close', attachment_satisfies=True)
        self._log_timing('grasp.close', started)
        started = time.monotonic()
        self._wait_attach()
        self._log_timing(
            'grasp.attach_confirm', started,
            'state={} model={}'.format(
                self.grasp_state, self.attached_model))
        # grasp_attach echoes the threshold-crossing joint position back to
        # the controller; reassert the configured close target for transport.
        started = time.monotonic()
        self._command_gripper(
            self.gripper_close, 'hold', attachment_satisfies=True)
        self._wait_attached_gripper_settle()
        self._log_timing('grasp.hold', started)
        # Reverse the verified grasp-approach trajectory before transport.
        # This avoids introducing a new wrist-folding motion immediately after
        # the cube has been attached.
        self._set_state('LIFT_GRASP')
        started = time.monotonic()
        self._move_arm('grasp_lift')
        self._check_attachment()
        self._log_timing('grasp.lift', started)
        self._log_timing('grasp.total', grasp_started)

    def _parking_footprint_inside(self, x, y, yaw, area):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        margin = self.parking_footprint_margin
        x_min = float(area['x_min']) + margin
        x_max = float(area['x_max']) - margin
        y_min = float(area['y_min']) + margin
        y_max = float(area['y_max']) - margin
        for vertex_x, vertex_y in self.parking_footprint:
            point_x = x + cos_yaw * float(vertex_x) - sin_yaw * float(vertex_y)
            point_y = y + sin_yaw * float(vertex_x) + cos_yaw * float(vertex_y)
            if not (x_min <= point_x <= x_max and y_min <= point_y <= y_max):
                return False
        return True

    def _fine_align_to_parking(self, target_x, target_y, target_yaw, area):
        self._set_state('FINE_ALIGN_TO_PARKING')
        deadline = time.monotonic() + self.parking_timeout
        period = 1.0 / self.fine_align_rate
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._check_attachment()
            base_x, base_y, base_yaw = self._base_pose_map()
            dx = target_x - base_x
            dy = target_y - base_y
            yaw_error = math.atan2(
                math.sin(target_yaw - base_yaw),
                math.cos(target_yaw - base_yaw))
            if (math.hypot(dx, dy) <= self.parking_position_tolerance
                    and abs(yaw_error) <= self.parking_yaw_tolerance
                    and self._parking_footprint_inside(
                        base_x, base_y, base_yaw, area)):
                self._stop_base()
                return

            cos_yaw = math.cos(base_yaw)
            sin_yaw = math.sin(base_yaw)
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy
            command = Twist()
            command.linear.x = self._clamp(
                self.fine_align_linear_gain * local_x,
                self.fine_align_max_linear_speed)
            command.linear.y = self._clamp(
                self.fine_align_linear_gain * local_y,
                self.fine_align_max_linear_speed)
            command.angular.z = self._clamp(
                self.fine_align_angular_gain * yaw_error,
                self.fine_align_max_angular_speed)
            self.cmd_vel_pub.publish(command)
            time.sleep(period)
        self._stop_base()
        raise RuntimeError('vehicle footprint did not settle inside parking area')

    def _navigate_to_parking(self):
        self._set_state('PREPARE_TRANSPORT')
        # Keep the attached object in the raised transport posture throughout.
        self._move_arm('transport')
        self._check_attachment()
        self._refresh_costmaps_after_pickup()
        self._set_state('NAVIGATE_TO_PARKING')
        area = self.parking_areas[self.target_category]
        target_x = 0.5 * (float(area['x_min']) + float(area['x_max']))
        target_y = 0.5 * (float(area['y_min']) + float(area['y_max']))
        for index, target_yaw in enumerate((0.0, math.pi)):
            if not self._navigate(
                    target_x, target_y, target_yaw,
                    'parking approach {}'.format(index),
                    retries=self.nav_retries):
                continue
            try:
                self._fine_align_to_parking(
                    target_x, target_y, target_yaw, area)
                self._check_attachment()
                self._set_state('PARKED')
                return
            except RuntimeError as exc:
                if (self.grasp_state != 'GRASPING'
                        or self.attached_model != self.category_to_model[
                            self.target_category]):
                    raise
                rospy.logwarn('parking alignment failed: %s', exc)
        raise RuntimeError('no parking approach goal was reachable')

    def run(self, initial_detection=None, failure_local_detection=False):
        self._set_state('INIT_VALIDATE')
        self._prepare_task_start()
        # This flag distinguishes a failed alignment/attachment from a later
        # transport or parking failure in the outer recovery loop.
        self.grasp_completed = False
        # Search first. Geometry measurement moves through grasp/place poses
        # and must not delay the initial navigation.
        detection = initial_detection
        if detection is None:
            detection = self._search()
            failure_local_detection = False
        self.target_was_located = True
        if any(value is None for value in (
                self.grasp_tcp_in_base, self.transport_tcp_in_base,
                self.place_tcp_in_base)):
            self._measure_arm_geometry()
        self._align_prepare_and_grasp(
            detection, require_source_area=not failure_local_detection)
        self.grasp_completed = True
        self._navigate_to_parking()
        self._set_state('SUCCESS')
        rospy.set_param('/gazebo_success', 1)
        self.result_pub.publish(String(data='SUCCESS:' + self.target_category))
        rospy.loginfo('pick-park succeeded for %s', self.target_category)


def main():
    rospy.init_node('pick_place_executor')
    try:
        executor = PickPlaceExecutor()
    except (rospy.ROSException, rospy.ROSInitException) as exc:
        rospy.logfatal('pick-place executor initialization failed: %s', exc)
        raise SystemExit(1)

    executor.wait_for_target_category()
    retry_detection = None
    retry_is_failure_local = False
    while not rospy.is_shutdown():
        try:
            executor.run(
                initial_detection=retry_detection,
                failure_local_detection=retry_is_failure_local)
            return
        except (rospy.ROSException, RuntimeError) as exc:
            rospy.logerr('pick-park attempt failed; keeping category %s: %s',
                         executor.target_category, exc)
            rospy.set_param('/gazebo_success', 0)
            executor.result_pub.publish(String(data='RETRYING:' + str(exc)))
            retry_detection = None
            retry_is_failure_local = False
            try:
                failure_pose = executor._base_pose_map()
            except RuntimeError as pose_exc:
                failure_pose = None
                rospy.logerr('could not record failure pose: %s', pose_exc)

            # Once a target has been located, keep recovering and searching
            # around the failure pose until that same category is seen again.
            # Do not clear target_category or return to WAITING_FOR_CATEGORY.
            recovery_failures = 0
            while not rospy.is_shutdown() and retry_detection is None:
                try:
                    recovered = executor._recover_after_failure()
                except (rospy.ROSException, RuntimeError) as recovery_exc:
                    recovered = False
                    rospy.logerr(
                        'pick-place failure recovery failed: %s', recovery_exc)
                if not recovered:
                    recovery_failures += 1
                    if recovery_failures >= MAX_FAILURE_RECOVERY_ATTEMPTS:
                        reason = (
                            'failure recovery did not complete after {} '
                            'attempts'.format(recovery_failures))
                        executor._stop_base()
                        executor._set_state('FAILED')
                        executor.result_pub.publish(
                            String(data='FAILED:' + reason))
                        rospy.logerr('%s; stopping task safely', reason)
                        return
                    time.sleep(executor.failure_search_retry_delay)
                    continue
                recovery_failures = 0
                if not executor.target_was_located:
                    # No target pose existed (for example, all source views
                    # missed), so retry the normal A/B/C search.
                    break
                if not executor.grasp_completed:
                    # Alignment, close, attachment confirmation, or lift
                    # failed. The failed pose is no longer trustworthy, so
                    # restart from the complete coarse search rather than
                    # repeatedly circling the same failed station.
                    rospy.logwarn(
                        'grasp did not complete for %s; restarting coarse '
                        'search', executor.target_category)
                    break
                if failure_pose is None:
                    try:
                        failure_pose = executor._base_pose_map()
                    except RuntimeError as pose_exc:
                        rospy.logerr(
                            'failure pose remains unavailable: %s', pose_exc)
                        time.sleep(executor.failure_search_retry_delay)
                        continue
                try:
                    retry_detection = executor._search_near_failure(
                        failure_pose)
                except (rospy.ROSException, RuntimeError) as search_exc:
                    rospy.logerr(
                        'failure-local target search failed: %s', search_exc)
                if retry_detection is None:
                    rospy.logwarn(
                        'target %s not found around failure pose; retrying '
                        'locally without waiting for another command',
                        executor.target_category)
                    time.sleep(executor.failure_search_retry_delay)
            retry_is_failure_local = retry_detection is not None


if __name__ == '__main__':
    main()
