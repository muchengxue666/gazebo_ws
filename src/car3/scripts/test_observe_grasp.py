#!/usr/bin/env python3
import math

import actionlib
import rospy
import tf2_geometry_msgs
import tf2_ros
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float64, String
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = [
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4', 'arm_joint5'
]
CATEGORY_TO_MODEL = {
    'food': 'cube_0',
    'electronics': 'cube_2',
}


class ObserveGraspTest:
    def __init__(self):
        self.target_category = rospy.get_param('~target_category', 'food').lower()
        if self.target_category not in CATEGORY_TO_MODEL:
            raise rospy.ROSInitException(
                'target_category must be food or electronics')

        self.min_confidence = float(rospy.get_param('~min_confidence', 0.38))
        self.max_pose_age = float(rospy.get_param('~max_pose_age', 0.8))
        self.detection_timeout = float(rospy.get_param('~detection_timeout', 20.0))
        self.arm_timeout = float(rospy.get_param('~arm_timeout', 8.0))
        self.gripper_timeout = float(rospy.get_param('~gripper_timeout', 3.0))
        self.attach_timeout = float(rospy.get_param('~attach_timeout', 5.0))
        self.joint_tolerance = float(rospy.get_param('~joint_tolerance', 0.03))
        self.goal_time_tolerance = float(
            rospy.get_param('~goal_time_tolerance', 1.5))
        self.settle_position_delta = float(
            rospy.get_param('~settle_position_delta', 0.002))
        self.settle_timeout = float(rospy.get_param('~settle_timeout', 2.0))
        self.settle_samples = int(rospy.get_param('~settle_samples', 5))
        self.ready_timeout = float(rospy.get_param('~ready_timeout', 2.0))

        calibration = rospy.get_param('/pick_place/calibration', {})
        self.arm_poses = calibration.get('arm_poses', {})
        self.gripper_open = calibration.get('gripper_open', 'UNSET')
        self.gripper_close = calibration.get('gripper_close', 'UNSET')
        self.observe = self._read_arm_pose('observe')
        self.grasp = self._read_arm_pose('grasp')
        self._validate_gripper(self.gripper_open, 'gripper_open')
        self._validate_gripper(self.gripper_close, 'gripper_close')

        self.joint_values = {}
        self.joint_velocities = {}
        self.category = 'unknown'
        self.confidence = 0.0
        self.pose = None
        self.detection_start_stamp = rospy.Time(0)
        self.ready = False
        self.state = 'IDLE'
        self.attached_model = ''

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.arm_client = actionlib.SimpleActionClient(
            '/arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        self.gripper_pub = rospy.Publisher(
            '/gripper_controller/command', Float64, queue_size=1)

        rospy.Subscriber('/joint_states', JointState, self._joint_cb)
        rospy.Subscriber('/cube_vision/category', String,
                         lambda msg: setattr(self, 'category', msg.data.lower()))
        rospy.Subscriber('/cube_vision/confidence', Float32,
                         lambda msg: setattr(self, 'confidence', float(msg.data)))
        rospy.Subscriber('/grasp_attach/ready', Bool,
                         lambda msg: setattr(self, 'ready', bool(msg.data)))
        rospy.Subscriber('/grasp_attach/state', String,
                         lambda msg: setattr(self, 'state', msg.data))
        rospy.Subscriber('/grasp_attach/attached_model', String,
                         lambda msg: setattr(self, 'attached_model', msg.data))

        # PoseStamped is used by cube_vision. Keep this import local so startup
        # errors clearly identify a mismatched vision topic type.
        from geometry_msgs.msg import PoseStamped
        rospy.Subscriber('/cube_vision/pose', PoseStamped, self._pose_cb)

    def _pose_cb(self, msg):
        self.pose = msg

    def _joint_cb(self, msg):
        values = dict(zip(msg.name, msg.position))
        self.joint_values.update({name: values[name] for name in JOINT_NAMES
                                  if name in values})
        velocities = dict(zip(msg.name, msg.velocity))
        self.joint_velocities.update({name: velocities.get(name, 0.0)
                                      for name in JOINT_NAMES
                                      if name in values})
        if 'r_joint' in values:
            self.joint_values['r_joint'] = values['r_joint']
            self.joint_velocities['r_joint'] = velocities.get('r_joint', 0.0)

    def _read_arm_pose(self, name):
        pose = self.arm_poses.get(name)
        if not isinstance(pose, dict):
            raise rospy.ROSInitException('missing calibration arm pose: ' + name)
        positions = pose.get('positions')
        duration = pose.get('duration')
        if not isinstance(positions, list) or len(positions) != len(JOINT_NAMES):
            raise rospy.ROSInitException(name + ' must contain five positions')
        if duration is None or float(duration) <= 0.0:
            raise rospy.ROSInitException(name + ' duration must be positive')
        values = [float(value) for value in positions]
        if not all(math.isfinite(value) and -3.14 <= value <= 3.14
                   for value in values):
            raise rospy.ROSInitException(name + ' contains an invalid joint value')
        return values, float(duration)

    @staticmethod
    def _validate_gripper(value, name):
        if isinstance(value, str) and value == 'UNSET':
            raise rospy.ROSInitException(name + ' is UNSET')
        value = float(value)
        if not math.isfinite(value) or not -1.51 <= value <= 1.51:
            raise rospy.ROSInitException(name + ' is outside r_joint limits')

    def _wait_joint_state(self, timeout):
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if all(name in self.joint_values for name in JOINT_NAMES):
                return True
            rospy.sleep(0.05)
        return False

    def _move_arm(self, pose_data, label):
        positions, duration = pose_data
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.05)
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = rospy.Duration(self.goal_time_tolerance)
        goal.goal_tolerance = [
            JointTolerance(name=name,
                           position=self.joint_tolerance,
                           velocity=-1.0,
                           acceleration=-1.0)
            for name in JOINT_NAMES
        ]
        # The controller's realtime path checks can abort while the position
        # is still converging. Disable those optional checks here and use the
        # measured joint position/velocity settling check below instead.
        goal.path_tolerance = [
            JointTolerance(name=name, position=-1.0,
                           velocity=-1.0, acceleration=-1.0)
            for name in JOINT_NAMES
        ]

        self.arm_client.send_goal(goal)
        if not self.arm_client.wait_for_result(rospy.Duration(self.arm_timeout)):
            self.arm_client.cancel_goal()
            raise RuntimeError(label + ' arm action timeout')
        state = self.arm_client.get_state()
        if state != GoalStatus.SUCCEEDED:
            result = self.arm_client.get_result()
            error_code = getattr(result, 'error_code', 'unknown')
            error_string = getattr(result, 'error_string', '')
            if error_code not in (-4, -5):
                raise RuntimeError(
                    '{} arm action failed: state={}, error_code={}, error={}'.format(
                        label, state, error_code, error_string))
            rospy.logwarn(
                '%s action reported goal tolerance failure; checking measured '
                'position and settling before deciding', label)

        if not self._wait_joint_state(1.0):
            raise RuntimeError(label + ' joint state unavailable')
        deadline = rospy.Time.now() + rospy.Duration(self.settle_timeout)
        settled = 0
        previous = None
        max_position_error = float('inf')
        max_position_delta = float('inf')
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            current = [self.joint_values[name] for name in JOINT_NAMES]
            errors = [abs(value - target)
                      for value, target in zip(current, positions)]
            max_position_error = max(errors)
            if previous is None:
                max_position_delta = float('inf')
            else:
                max_position_delta = max(abs(value - old)
                                         for value, old in zip(current, previous))
            if (max_position_error <= self.joint_tolerance
                    and max_position_delta <= self.settle_position_delta):
                settled += 1
                if settled >= self.settle_samples:
                    if state != GoalStatus.SUCCEEDED:
                        rospy.logwarn(
                            '%s accepted after measured settling: position_error=%.6f '
                            'position_delta=%.6f', label, max_position_error,
                            max_position_delta)
                    return
            else:
                settled = 0
            previous = current
            rospy.sleep(0.05)
        raise RuntimeError(
            '{} arm action not settled: position_error={:.6f}, '
            'position_delta={:.6f}'.format(
                label, max_position_error, max_position_delta))

    def _command_gripper(self, value, label, predicate):
        self.gripper_pub.publish(Float64(data=float(value)))
        deadline = rospy.Time.now() + rospy.Duration(self.gripper_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            position = self.joint_values.get('r_joint')
            if position is not None and predicate(position):
                return
            rospy.sleep(0.05)
        raise RuntimeError(label + ' gripper position timeout')

    def _wait_detection(self, since_stamp):
        del since_stamp
        deadline = rospy.Time.now() + rospy.Duration(self.detection_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            pose = self.pose
            if pose is not None and self.category == self.target_category:
                age = (rospy.Time.now() - pose.header.stamp).to_sec()
                if (0.0 <= age <= self.max_pose_age
                        and self.confidence >= self.min_confidence
                        and pose.header.frame_id):
                    return pose
            rospy.sleep(0.05)
        pose_stamp = self.pose.header.stamp.to_sec() if self.pose else 0.0
        pose_frame = self.pose.header.frame_id if self.pose else ''
        pose_age = ((rospy.Time.now() - self.pose.header.stamp).to_sec()
                    if self.pose else float('inf'))
        raise RuntimeError(
            'target vision detection timeout: category={}, confidence={:.3f}, '
            'pose_frame={}, pose_age={:.3f}s, pose_stamp={:.3f}'.format(
                self.category, self.confidence, pose_frame, pose_age, pose_stamp))

    def _transform_pose_to_base(self, pose):
        if not pose.header.frame_id:
            raise RuntimeError('vision pose has no frame')
        point = PointStamped()
        point.header = pose.header
        point.point = pose.pose.position
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint', pose.header.frame_id, pose.header.stamp,
                rospy.Duration(0.2))
            return tf2_geometry_msgs.do_transform_point(point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            raise RuntimeError('vision pose TF unavailable: {}'.format(exc))

    def _wait_attach(self):
        expected = CATEGORY_TO_MODEL[self.target_category]
        deadline = rospy.Time.now() + rospy.Duration(self.attach_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.state == 'GRASPING' and self.attached_model == expected:
                return
            rospy.sleep(0.05)
        raise RuntimeError(
            'attach timeout: state={}, model={}'.format(
                self.state, self.attached_model))

    def run(self):
        rospy.loginfo('observe-to-grasp test target=%s', self.target_category)
        if not self.arm_client.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError('arm trajectory action server unavailable')
        if not self._wait_joint_state(5.0):
            raise RuntimeError('joint state unavailable')

        # Always open first. This also releases an attachment left by a prior
        # manual trial before the next observation starts.
        self._command_gripper(
            self.gripper_open, 'open', lambda value: value > 0.89)
        if self.state != 'IDLE' or self.attached_model:
            raise RuntimeError('grasp_attach is not idle after opening gripper')

        detection_start = self.pose.header.stamp if self.pose is not None else rospy.Time(0)
        self._move_arm(self.observe, 'observe')
        detected = self._wait_detection(detection_start)
        point_base = self._transform_pose_to_base(detected)
        rospy.loginfo(
            'target detected: category=%s confidence=%.3f base=(%.3f, %.3f, %.3f)',
            self.category, self.confidence, point_base.point.x,
            point_base.point.y, point_base.point.z)

        # No cmd_vel is issued in this first test. The log is the calibration
        # input for the later base alignment stage.
        self._move_arm(self.grasp, 'grasp')
        if not self.ready:
            deadline = rospy.Time.now() + rospy.Duration(self.ready_timeout)
            while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                if self.ready:
                    break
                rospy.sleep(0.05)
        if not self.ready:
            raise RuntimeError('grasp_attach ready was not true at grasp pose')

        self._command_gripper(
            self.gripper_close, 'close', lambda value: value < 0.8)
        self._wait_attach()
        rospy.loginfo('observe-to-grasp succeeded; holding object')


def main():
    rospy.init_node('test_observe_grasp')
    try:
        ObserveGraspTest().run()
    except (rospy.ROSException, rospy.ROSInitException, RuntimeError) as exc:
        rospy.logerr('observe-to-grasp FAILED: %s', exc)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
