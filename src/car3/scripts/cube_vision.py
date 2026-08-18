#!/usr/bin/env python3
import collections
import os
import sys

import cv2
import message_filters
import numpy as np
import rospy
import tf.transformations as transformations
import tf2_geometry_msgs
import tf2_ros

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cube_vision_core import CubeVisionCore
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String
from std_srvs.srv import Empty, EmptyResponse


class DepthImageFilter(message_filters.SimpleFilter):
    """Forward only metric single-channel images from a mixed Gazebo topic."""

    VALID_ENCODINGS = frozenset(('32FC1', '16UC1', 'mono16'))

    def __init__(self, topic):
        super().__init__()
        self.subscriber = rospy.Subscriber(
            topic, Image, self._callback, queue_size=1,
            buff_size=640 * 480 * 16)

    def _callback(self, msg):
        if msg.encoding in self.VALID_ENCODINGS:
            self.signalMessage(msg)


class CubeVision:
    CATEGORIES = ('food', 'daily', 'electronics')

    def __init__(self):
        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.output_frame = rospy.get_param('~output_frame', 'map')
        self.camera_info_topic = rospy.get_param(
            '~camera_info_topic', '/camera/rgb/camera_info')
        self.template_dir = rospy.get_param('~template_dir')
        self.processing_rate = float(rospy.get_param('~processing_rate', 5.0))
        self.min_depth = float(rospy.get_param('~min_depth', 0.12))
        self.max_depth = float(rospy.get_param('~max_depth', 1.0))
        self.min_face_size = float(rospy.get_param('~min_face_size', 0.025))
        self.max_face_size = float(rospy.get_param('~max_face_size', 0.12))
        self.cube_half_size = float(rospy.get_param('~cube_half_size', 0.02))
        self.depth_plane_min_points = int(
            rospy.get_param('~depth_plane_min_points', 30))
        self.depth_plane_max_residual = float(
            rospy.get_param('~depth_plane_max_residual', 0.004))
        # The Gazebo depth sensor has an additional pi roll in its SDF pose,
        # while robot_state_publisher only exposes the URDF optical frame.
        # Convert points from the rendered sensor frame into that TF frame
        # before applying camera_depth_optical_frame -> output_frame.
        sensor_rpy = rospy.get_param(
            '~sensor_to_optical_rpy', [3.141592653589793, 0.0, 0.0])
        if not isinstance(sensor_rpy, (list, tuple)) or len(sensor_rpy) != 3:
            raise rospy.ROSInitException('sensor_to_optical_rpy must have 3 values')
        self.sensor_to_optical = np.asarray(
            transformations.euler_matrix(*[float(value) for value in sensor_rpy])[:3, :3],
            dtype=np.float64)
        self.stability_samples = int(rospy.get_param('~stability_samples', 5))
        self.required_votes = int(rospy.get_param('~required_votes', 3))
        self.max_position_std = float(rospy.get_param('~max_position_std', 0.025))
        self.max_unknown_frames = int(rospy.get_param('~max_unknown_frames', 1))
        self.publish_debug_image = bool(rospy.get_param('~publish_debug_image', True))
        self.depth_debug_max = float(rospy.get_param('~depth_debug_max', 4.0))
        self.rgb_pose_fallback = bool(rospy.get_param('~rgb_pose_fallback', True))
        self.rgb_cube_size = float(rospy.get_param('~rgb_cube_size', 0.04))
        self.rgb_pose_min_depth = float(rospy.get_param('~rgb_pose_min_depth', 0.05))
        self.rgb_pose_max_depth = float(rospy.get_param('~rgb_pose_max_depth', 0.8))

        core_defaults = (
            ('brightness_threshold', 80),
            ('brightness_offset', 20),
            ('brightness_blur', 0),
            ('min_candidate_area', 300.0),
            ('max_candidate_area', 30000.0),
            ('min_candidate_pixels', 20),
            ('max_candidate_pixels', 300),
            ('max_candidate_aspect', 2.5),
            ('min_candidate_fill', 0.45),
            ('min_candidate_solidity', 0.82),
            ('min_dark_fraction', 0.04),
            ('dark_pixel_offset', 18),
            ('roi_padding', 4),
            ('feature_scale', 3.0),
            ('feature_ratio', 0.74),
            ('feature_ransac_threshold', 7.0),
            ('min_feature_matches', 4),
            ('min_feature_inliers', 4),
            ('min_template_coverage', 0.008),
            ('min_roi_coverage', 0.004),
            ('min_category_margin', 0.45),
            ('template_category_fraction', 0.55),
        )
        core_params = {
            name: rospy.get_param('~' + name, default)
            for name, default in core_defaults
        }
        try:
            self.core = CubeVisionCore(self.template_dir, core_params)
        except RuntimeError as exc:
            raise rospy.ROSInitException(str(exc))

        self.history = collections.deque(maxlen=self.stability_samples)
        self.category_history = collections.deque(maxlen=self.stability_samples)
        self.unknown_frames = 0
        self.pose_unknown_frames = 0
        self.last_process_time = rospy.Time(0)
        self.last_diagnostic_time = rospy.Time(0)
        self.last_candidate_diagnostics = {}

        self.category_pub = rospy.Publisher('~category', String, queue_size=1)
        self.confidence_pub = rospy.Publisher('~confidence', Float32, queue_size=1)
        self.pose_pub = rospy.Publisher('~pose', PoseStamped, queue_size=1)
        self.debug_pub = rospy.Publisher('~debug_image', Image, queue_size=1)
        self.depth_debug_pub = rospy.Publisher('~depth_debug', Image, queue_size=1)
        self.reset_srv = rospy.Service('~reset', Empty, self._reset_cb)

        rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        depth_sub = DepthImageFilter('/depth_camera/depth/image_raw')
        rgb_info_sub = message_filters.Subscriber(
            self.camera_info_topic, CameraInfo)
        # Gazebo's depth-camera plugin in ROS Noetic does not publish its
        # depth CameraInfo topic. RGB and depth are co-located and configured
        # with the same image geometry in car3.urdf, so they share this model.
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, rgb_info_sub], queue_size=8, slop=0.08)
        self.sync.registerCallback(self._image_cb)

        rospy.loginfo(
            'cube_vision ready: frame=%s classifier=SIFT templates=%s',
            self.output_frame, ','.join(self.core.templates.keys()))

    def _reset_cb(self, _request):
        self.history.clear()
        self.category_history.clear()
        self.unknown_frames = 0
        self.pose_unknown_frames = 0
        return EmptyResponse()

    @staticmethod
    def _depth_in_meters(depth, encoding):
        if encoding in ('16UC1', 'mono16') or depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    def _publish_depth_debug(self, depth, header):
        if (not self.publish_debug_image
                or self.depth_debug_pub.get_num_connections() == 0):
            return
        finite = np.isfinite(depth)
        valid = finite & (depth > 0.02) & (depth < self.depth_debug_max)
        debug = np.zeros(depth.shape, dtype=np.uint8)
        debug[valid] = np.clip(
            255.0 * (self.depth_debug_max - depth[valid]) / self.depth_debug_max,
            0.0, 255.0).astype(np.uint8)
        msg = Image()
        msg.header = header
        msg.height = debug.shape[0]
        msg.width = debug.shape[1]
        msg.encoding = 'mono8'
        msg.is_bigendian = False
        msg.step = int(debug.strides[0])
        msg.data = debug.tobytes()
        self.depth_debug_pub.publish(msg)

    def _image_cb(self, rgb_msg, depth_msg, camera_info_msg):
        now = rospy.Time.now()
        if (now - self.last_process_time).to_sec() < 1.0 / self.processing_rate:
            return
        self.last_process_time = now

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            raw_depth = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            rospy.logwarn_throttle(
                2.0, 'cube_vision image conversion failed: %s', exc)
            return

        depth = self._depth_in_meters(raw_depth, depth_msg.encoding)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        self._publish_depth_debug(depth, depth_msg.header)
        if depth.ndim != 2:
            rospy.logwarn_throttle(
                2.0, 'unsupported depth image shape: %s', depth.shape)
            return
        if depth.shape[:2] != rgb.shape[:2]:
            rospy.logwarn_throttle(
                2.0, 'RGB/depth dimensions differ: %s vs %s',
                rgb.shape[:2], depth.shape[:2])
            return

        if (camera_info_msg.width != rgb.shape[1]
                or camera_info_msg.height != rgb.shape[0]):
            rospy.logwarn_throttle(
                2.0, 'CameraInfo/image dimensions differ: info=%dx%d image=%dx%d',
                camera_info_msg.width, camera_info_msg.height,
                rgb.shape[1], rgb.shape[0])
            return
        if (depth_msg.header.frame_id
                and camera_info_msg.header.frame_id
                and depth_msg.header.frame_id != camera_info_msg.header.frame_id):
            rospy.logwarn_throttle(
                2.0, 'RGB/depth frame mismatch: depth=%s info=%s',
                depth_msg.header.frame_id, camera_info_msg.header.frame_id)
            return
        self.camera_model.fromCameraInfo(camera_info_msg)
        candidates = self._find_candidates(rgb, depth)
        debug = (rgb.copy()
                 if (self.publish_debug_image
                     and self.debug_pub.get_num_connections() > 0)
                 else None)

        if not candidates:
            self._diagnose(now, 'no cube candidate')
            self._record_unknown_frame()
            self._publish_unknown(0.0)
            self._publish_debug(debug, rgb_msg)
            return

        best = candidates[0]
        valid_match = best['valid_match']
        if debug is not None:
            self._draw_candidate(debug, best, valid_match)
        if not valid_match:
            self._record_unknown_frame()
            self._publish_unknown(0.0)
            self._publish_debug(debug, rgb_msg)
            return

        self.unknown_frames = 0
        self.category_history.append((best['category'], best['score']))
        stable_category = self._stable_category_result()
        if stable_category is None:
            self._publish_unknown(best['score'])
            self._publish_debug(debug, rgb_msg)
            return
        category, confidence = stable_category
        self.category_pub.publish(String(data=category))
        self.confidence_pub.publish(Float32(data=confidence))

        if not best['position_valid']:
            if self.rgb_pose_fallback:
                fallback = self._rgb_fallback_pose(best, depth_msg.header)
                if fallback is not None:
                    self.history.append((best['category'], best['score'], fallback))
                    stable = self._stable_result()
                    if stable is not None:
                        self.pose_pub.publish(stable[2])
                    self._publish_debug(debug, rgb_msg)
                    return
            self._record_pose_unknown_frame()
            self._publish_debug(debug, rgb_msg)
            return

        pose = self._candidate_pose(best, depth_msg.header)
        if pose is None:
            self._record_pose_unknown_frame()
            self._publish_debug(debug, rgb_msg)
            return

        self.pose_unknown_frames = 0
        self.history.append((best['category'], best['score'], pose))
        stable = self._stable_result()
        if stable is not None:
            self.pose_pub.publish(stable[2])
        self._publish_debug(debug, rgb_msg)

    def _find_candidates(self, rgb, depth):
        candidates, _ = self.core.detect(rgb)
        diagnostics = collections.Counter(self.core.last_diagnostics)
        camera_matrix = np.asarray(
            self.camera_model.intrinsicMatrix(), dtype=np.float64)
        for candidate in candidates:
            depth_geometry = self.core.estimate_cube_center(
                depth, self.core.inset_mask(candidate, scale=0.82),
                camera_matrix, self.cube_half_size,
                self.min_depth, self.max_depth,
                self.depth_plane_min_points,
                self.depth_plane_max_residual)
            position_valid = False
            if depth_geometry is None:
                diagnostics['depth_plane_invalid'] += 1
                candidate['depth'] = float('nan')
                candidate['valid_depth_pixels'] = 0
                candidate['camera_center'] = None
            else:
                z = depth_geometry['depth']
                metric_width = candidate['width'] * z / float(self.camera_model.fx())
                metric_height = candidate['height'] * z / float(self.camera_model.fy())
                candidate['depth'] = z
                candidate['valid_depth_pixels'] = depth_geometry['valid_points']
                candidate['depth_residual'] = depth_geometry['residual_rms']
                candidate['camera_center'] = depth_geometry['center']
                if not (self.min_face_size <= min(metric_width, metric_height)
                        and max(metric_width, metric_height) <= self.max_face_size):
                    diagnostics['metric_size'] += 1
                else:
                    position_valid = True
                    diagnostics['position_valid'] += 1
            candidate['position_valid'] = position_valid

        self.last_candidate_diagnostics = dict(diagnostics)
        return candidates

    def _rgb_fallback_pose(self, candidate, header):
        quad = candidate['face_quad']
        if quad is None:
            return None
        widths = (np.linalg.norm(quad[1] - quad[0]),
                  np.linalg.norm(quad[2] - quad[3]))
        heights = (np.linalg.norm(quad[3] - quad[0]),
                   np.linalg.norm(quad[2] - quad[1]))
        pixels = max(min(0.5 * sum(widths), 0.5 * sum(heights)), 1.0)
        focal = 0.5 * (self.camera_model.fx() + self.camera_model.fy())
        z = focal * self.rgb_cube_size / pixels
        if not self.rgb_pose_min_depth <= z <= self.rgb_pose_max_depth:
            return None
        return self._pixel_pose(candidate['center'], z, header, 'RGB pose')

    def _candidate_pose(self, candidate, header):
        center = candidate['camera_center']
        if center is None:
            return None
        point = PointStamped()
        point.header = header
        corrected = self.sensor_to_optical.dot(np.asarray(center, dtype=np.float64))
        point.point.x = float(corrected[0])
        point.point.y = float(corrected[1])
        point.point.z = float(corrected[2])
        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame, header.frame_id, header.stamp,
                rospy.Duration(0.08))
            transformed = tf2_geometry_msgs.do_transform_point(point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0, 'cube_vision depth pose TF unavailable: %s', exc)
            return None
        pose = PoseStamped()
        pose.header = transformed.header
        pose.pose.position = transformed.point
        pose.pose.orientation.w = 1.0
        return pose

    def _pixel_pose(self, center, z, header, label):
        u, v = center
        ray = np.asarray(
            self.camera_model.projectPixelTo3dRay((float(u), float(v))),
            dtype=np.float64)
        if abs(ray[2]) < 1e-6:
            return None
        point = PointStamped()
        point.header = header
        corrected = self.sensor_to_optical.dot(np.asarray(
            [ray[0] * z / ray[2], ray[1] * z / ray[2], z], dtype=np.float64))
        point.point.x = float(corrected[0])
        point.point.y = float(corrected[1])
        point.point.z = float(corrected[2])
        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame, header.frame_id, header.stamp,
                rospy.Duration(0.08))
            transformed = tf2_geometry_msgs.do_transform_point(point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0, 'cube_vision %s TF unavailable: %s', label, exc)
            return None
        pose = PoseStamped()
        pose.header = transformed.header
        pose.pose.position = transformed.point
        pose.pose.orientation.w = 1.0
        return pose

    def _stable_category_result(self):
        if len(self.category_history) < self.required_votes:
            return None
        counts = collections.Counter(item[0] for item in self.category_history)
        category, votes = counts.most_common(1)[0]
        if votes < self.required_votes:
            return None
        scores = [score for name, score in self.category_history
                  if name == category]
        return category, float(np.mean(scores))

    def _stable_result(self):
        if len(self.history) < self.required_votes:
            return None
        counts = collections.Counter(item[0] for item in self.history)
        category, votes = counts.most_common(1)[0]
        if votes < self.required_votes:
            return None
        matching = [item for item in self.history if item[0] == category]
        positions = np.array([
            [item[2].pose.position.x,
             item[2].pose.position.y,
             item[2].pose.position.z]
            for item in matching
        ])
        if np.max(np.std(positions, axis=0)) > self.max_position_std:
            return None

        latest = matching[-1][2]
        stable_pose = PoseStamped()
        stable_pose.header = latest.header
        stable_pose.pose.position.x = float(np.mean(positions[:, 0]))
        stable_pose.pose.position.y = float(np.mean(positions[:, 1]))
        stable_pose.pose.position.z = float(np.mean(positions[:, 2]))
        stable_pose.pose.orientation.w = 1.0
        confidence = float(np.mean([item[1] for item in matching]))
        return category, confidence, stable_pose

    def _record_unknown_frame(self):
        self.unknown_frames += 1
        if self.unknown_frames > self.max_unknown_frames:
            self.history.clear()
            self.category_history.clear()
            self.pose_unknown_frames = 0

    def _record_pose_unknown_frame(self):
        self.pose_unknown_frames += 1
        if self.pose_unknown_frames > self.max_unknown_frames:
            self.history.clear()

    def _publish_unknown(self, confidence):
        self.category_pub.publish(String(data='unknown'))
        self.confidence_pub.publish(
            Float32(data=max(0.0, float(confidence))))

    def _diagnose(self, now, reason):
        if (now - self.last_diagnostic_time).to_sec() > 2.0:
            rospy.logwarn(
                'cube_vision: %s diagnostics=%s',
                reason, self.last_candidate_diagnostics)
            self.last_diagnostic_time = now

    @staticmethod
    def _draw_candidate(image, candidate, valid):
        color = (0, 200, 0) if valid else (0, 165, 255)
        hull = np.round(candidate['hull']).astype(np.int32)
        cv2.polylines(image, [hull], True, color, 2)
        face_quad = candidate['face_quad']
        if valid and face_quad is not None:
            cv2.polylines(
                image, [np.round(face_quad).astype(np.int32)],
                True, (255, 200, 0), 1)
        x, y = np.round(candidate['center']).astype(int)
        depth_text = ('{:.2f}m'.format(candidate['depth'])
                      if np.isfinite(candidate['depth']) else 'no-depth')
        details = candidate['match_details'][candidate['category']]
        label = '{} {:.2f} {}/{} {}'.format(
            candidate['category'], candidate['score'], details['inliers'],
            details['matches'], depth_text)
        cv2.putText(image, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _publish_debug(self, image, source_msg):
        if image is None:
            return
        try:
            msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            msg.header = source_msg.header
            self.debug_pub.publish(msg)
        except CvBridgeError:
            pass


if __name__ == '__main__':
    rospy.init_node('cube_vision')
    CubeVision()
    rospy.spin()
