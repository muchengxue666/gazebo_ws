#!/usr/bin/env python3
import math
import time

import rospy
import tf.transformations as transformations
import tf2_ros
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import PoseStamped


class VisionGroundTruthReport:
    def __init__(self):
        self.target_category = rospy.get_param('~target_category', 'food').lower()
        self.category_to_model = rospy.get_param('~category_to_model', {
            'food': 'cube_0', 'daily': 'cube_1', 'electronics': 'cube_2'})
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.wait_for_service('/gazebo/get_model_state', timeout=10.0)
        self.get_model_state = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)
        self.vision_pose = None
        rospy.Subscriber('/cube_vision/pose', PoseStamped, self._pose_cb)

    def _pose_cb(self, msg):
        self.vision_pose = msg

    @staticmethod
    def _se2_from_xyyaw(x, y, yaw):
        return x, y, yaw

    @staticmethod
    def _compose(a, b):
        ax, ay, ayaw = a
        bx, by, byaw = b
        c = math.cos(ayaw)
        s = math.sin(ayaw)
        return (ax + c * bx - s * by,
                ay + s * bx + c * by,
                math.atan2(math.sin(ayaw + byaw),
                            math.cos(ayaw + byaw)))

    @staticmethod
    def _inverse(t):
        x, y, yaw = t
        c = math.cos(yaw)
        s = math.sin(yaw)
        return (-c * x - s * y, s * x - c * y, -yaw)

    @staticmethod
    def _pose_xyyaw(pose):
        if hasattr(pose, 'position'):
            position = pose.position
            orientation = pose.orientation
        else:
            position = pose.translation
            orientation = pose.rotation
        q = [orientation.x, orientation.y,
             orientation.z, orientation.w]
        return (position.x, position.y,
                transformations.euler_from_quaternion(q)[2])

    def sample(self):
        robot = self.get_model_state('car3', 'world')
        cube = self.get_model_state(
            self.category_to_model[self.target_category], 'world')
        if not robot.success or not cube.success:
            raise RuntimeError('Gazebo model state unavailable')
        try:
            map_base_tf = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rospy.Time(0), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            raise RuntimeError('map/base TF unavailable: {}'.format(exc))

        map_base = self._pose_xyyaw(map_base_tf.transform)
        world_base = self._pose_xyyaw(robot.pose)
        map_world = self._compose(map_base, self._inverse(world_base))
        map_world_z = (map_base_tf.transform.translation.z
                       - robot.pose.position.z)
        cube_world = self._pose_xyyaw(cube.pose)
        cube_map_gt = self._compose(map_world, cube_world)
        cube_map_gt_z = map_world_z + cube.pose.position.z
        report = {
            'map_world': map_world,
            'map_world_z': map_world_z,
            'cube_world': cube_world,
            'cube_map_gt': cube_map_gt,
            'cube_map_gt_z': cube_map_gt_z,
        }
        if self.vision_pose is not None:
            vision = self.vision_pose.pose.position
            report['vision_map'] = (vision.x, vision.y, vision.z)
            report['xy_error'] = math.hypot(
                vision.x - cube_map_gt[0], vision.y - cube_map_gt[1])
            report['z_error'] = vision.z - cube_map_gt_z
        return report

    def run(self):
        rospy.loginfo('Waiting for a stationary vision pose for %s', self.target_category)
        deadline = time.monotonic() + rospy.get_param('~timeout', 30.0)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.vision_pose is not None:
                report = self.sample()
                rospy.loginfo('vision ground truth: %s', report)
                return
            rospy.sleep(0.1)
        raise RuntimeError('vision pose timeout')


if __name__ == '__main__':
    rospy.init_node('calibrate_vision_ground_truth')
    try:
        VisionGroundTruthReport().run()
    except (rospy.ROSException, RuntimeError) as exc:
        rospy.logerr('vision ground truth failed: %s', exc)
        raise SystemExit(1)
