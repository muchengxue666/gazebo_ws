#!/usr/bin/env python3
import collections
import os

import cv2
import message_filters
import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String


def _order_quad(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


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
        self.min_face_pixels = int(rospy.get_param('~min_face_pixels', 18))
        self.max_face_pixels = int(rospy.get_param('~max_face_pixels', 180))
        self.min_quad_area = float(rospy.get_param('~min_quad_area', 260.0))
        self.min_face_size = float(rospy.get_param('~min_face_size', 0.025))
        self.max_face_size = float(rospy.get_param('~max_face_size', 0.075))
        self.template_score_threshold = float(
            rospy.get_param('~template_score_threshold', 0.38))
        self.template_score_margin = float(
            rospy.get_param('~template_score_margin', 0.04))
        self.surface_to_center_depth = float(
            rospy.get_param('~surface_to_center_depth', 0.02))
        self.stability_samples = int(rospy.get_param('~stability_samples', 5))
        self.required_votes = int(rospy.get_param('~required_votes', 3))
        self.max_position_std = float(rospy.get_param('~max_position_std', 0.025))
        self.publish_debug_image = bool(rospy.get_param('~publish_debug_image', True))
        self.depth_debug_max = float(rospy.get_param('~depth_debug_max', 4.0))
        self.rgb_pose_fallback = bool(rospy.get_param('~rgb_pose_fallback', True))
        self.rgb_cube_size = float(rospy.get_param('~rgb_cube_size', 0.04))
        self.rgb_pose_min_depth = float(rospy.get_param('~rgb_pose_min_depth', 0.05))
        self.rgb_pose_max_depth = float(rospy.get_param('~rgb_pose_max_depth', 0.8))

        self.templates = self._load_templates()
        self.history = collections.deque(maxlen=self.stability_samples)
        self.category_history = collections.deque(maxlen=self.stability_samples)
        self.last_process_time = rospy.Time(0)
        self.last_diagnostic_time = rospy.Time(0)
        self.last_candidate_diagnostics = {}

        self.category_pub = rospy.Publisher('~category', String, queue_size=1)
        self.confidence_pub = rospy.Publisher('~confidence', Float32, queue_size=1)
        self.pose_pub = rospy.Publisher('~pose', PoseStamped, queue_size=1)
        self.debug_pub = rospy.Publisher('~debug_image', Image, queue_size=1)
        self.depth_debug_pub = rospy.Publisher('~depth_debug', Image, queue_size=1)

        rgb_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('/depth_camera/depth/image_raw', Image)
        info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=8, slop=0.08)
        self.sync.registerCallback(self._image_cb)

        rospy.loginfo(
            'cube_vision ready: frame=%s templates=%s',
            self.output_frame, ','.join(self.templates.keys()))

    def _load_templates(self):
        files = {
            'food': 'Food.png',
            'daily': 'Daily_Necessities.png',
            'electronics': 'Electronics.png',
        }
        templates = {}
        for category, filename in files.items():
            path = os.path.join(self.template_dir, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise rospy.ROSInitException(
                    'Cannot load cube template: {}'.format(path))
            templates[category] = self._prepare_patch(image)
        return templates

    @staticmethod
    def _prepare_patch(image):
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        image = cv2.equalizeHist(image)
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 7)
        return cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    @staticmethod
    def _depth_in_meters(depth, encoding):
        if encoding in ('16UC1', 'mono16') or depth.dtype == np.uint16:
            return depth.astype(np.float32) * 0.001
        return depth.astype(np.float32)

    def _publish_depth_debug(self, depth, header):
        finite = np.isfinite(depth)
        valid = finite & (depth > 0.02) & (depth < self.depth_debug_max)
        debug = np.zeros(depth.shape, dtype=np.uint8)
        debug[valid] = np.clip(
            255.0 * (self.depth_debug_max - depth[valid]) / self.depth_debug_max,
            0.0, 255.0).astype(np.uint8)
        try:
            msg = Image()
            msg.header = header
            msg.height = debug.shape[0]
            msg.width = debug.shape[1]
            msg.encoding = 'mono8'
            msg.is_bigendian = False
            msg.step = int(debug.strides[0])
            msg.data = debug.tobytes()
            self.depth_debug_pub.publish(msg)
        except CvBridgeError:
            pass

    def _image_cb(self, rgb_msg, depth_msg, info_msg):
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
        self._publish_depth_debug(depth, depth_msg.header)
        if depth.shape[:2] != rgb.shape[:2]:
            rospy.logwarn_throttle(
                2.0, 'RGB/depth dimensions differ: %s vs %s',
                rgb.shape[:2], depth.shape[:2])
            return

        self.camera_model.fromCameraInfo(info_msg)
        candidates = self._find_candidates(rgb, depth)
        debug = rgb.copy()

        if not candidates:
            if (now - self.last_diagnostic_time).to_sec() > 2.0:
                rospy.logwarn('cube_vision: no valid cube candidate after geometry/depth filters')
                self.last_diagnostic_time = now
            self.history.clear()
            self.category_history.clear()
            self._publish_unknown(0.0)
            self._publish_debug(debug, rgb_msg)
            return

        candidates.sort(key=lambda item: item['score'], reverse=True)
        best = candidates[0]
        category_scores = sorted(
            best['category_scores'].values(), reverse=True)
        category_margin = category_scores[0] - category_scores[1]
        valid_match = (
            best['score'] >= self.template_score_threshold
            and category_margin >= self.template_score_margin)

        self._draw_candidate(debug, best, valid_match)
        if not valid_match:
            self.history.clear()
            self.category_history.clear()
            self._publish_unknown(best['score'])
            self._publish_debug(debug, rgb_msg)
            return

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
                fallback = self._rgb_fallback_pose(best, info_msg.header)
                if fallback is not None:
                    self.history.append((best['category'], best['score'], fallback))
                    stable = self._stable_result()
                    if stable is not None:
                        _, _, stable_pose = stable
                        self.pose_pub.publish(stable_pose)
                    self._publish_debug(debug, rgb_msg)
                    return
            self.history.clear()
            self._publish_debug(debug, rgb_msg)
            return

        pose = self._candidate_pose(best, info_msg.header)
        if pose is None:
            self.history.clear()
            self._publish_debug(debug, rgb_msg)
            return

        self.history.append((best['category'], best['score'], pose))
        stable = self._stable_result()
        if stable is not None:
            _, _, stable_pose = stable
            self.pose_pub.publish(stable_pose)
        self._publish_debug(debug, rgb_msg)

    def _find_candidates(self, rgb, depth):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        diagnostics = collections.Counter()
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 135)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            diagnostics['quads'] += 1
            area = abs(cv2.contourArea(polygon))
            if area < self.min_quad_area:
                diagnostics['small_area'] += 1
                continue

            quad = _order_quad(polygon.reshape(4, 2))
            widths = [np.linalg.norm(quad[1] - quad[0]),
                      np.linalg.norm(quad[2] - quad[3])]
            heights = [np.linalg.norm(quad[3] - quad[0]),
                       np.linalg.norm(quad[2] - quad[1])]
            width = float(sum(widths) * 0.5)
            height = float(sum(heights) * 0.5)
            short_side = min(width, height)
            long_side = max(width, height)
            if (short_side < self.min_face_pixels
                    or long_side > self.max_face_pixels
                    or long_side / max(short_side, 1.0) > 2.1):
                diagnostics['pixel_size'] += 1
                continue

            center = np.mean(quad, axis=0)
            if self._duplicate_center(candidates, center, short_side * 0.4):
                continue

            # Category recognition only requires the RGB quadrilateral. Depth
            # is evaluated separately below and gates pose publication only.
            patch = self._warp_patch(gray, quad)
            category, score, category_scores = self._classify_patch(patch)

            z = self._median_depth(depth, quad, center)
            position_valid = False
            if z is None:
                diagnostics['depth_invalid'] += 1
                z = float('nan')
            else:
                fx = float(self.camera_model.fx())
                fy = float(self.camera_model.fy())
                metric_width = width * z / fx
                metric_height = height * z / fy
                if z >= self.max_depth * 0.98:
                    diagnostics['depth_at_config_limit'] += 1
                elif not (self.min_face_size <= min(metric_width, metric_height)
                          and max(metric_width, metric_height) <= self.max_face_size):
                    diagnostics['metric_size'] += 1
                else:
                    position_valid = True
                    diagnostics['position_valid'] += 1

            candidates.append({
                'quad': quad,
                'center': center,
                'depth': z,
                'position_valid': position_valid,
                'category': category,
                'score': score,
                'category_scores': category_scores,
            })
        self.last_candidate_diagnostics = dict(diagnostics)
        if not candidates and (rospy.Time.now() - self.last_diagnostic_time).to_sec() > 2.0:
            rospy.logwarn('cube_vision: candidate diagnostics=%s', self.last_candidate_diagnostics)
            self.last_diagnostic_time = rospy.Time.now()
        return candidates

    @staticmethod
    def _duplicate_center(candidates, center, threshold):
        return any(np.linalg.norm(item['center'] - center) < threshold
                   for item in candidates)

    @staticmethod
    def _warp_patch(gray, quad):
        target = np.array(
            [[0, 0], [127, 0], [127, 127], [0, 127]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(quad, target)
        return cv2.warpPerspective(gray, transform, (128, 128))

    def _classify_patch(self, patch):
        prepared = self._prepare_patch(patch)
        scores = {}
        for category, template in self.templates.items():
            variants = []
            rotated = prepared
            for _ in range(4):
                variants.append(rotated)
                variants.append(cv2.flip(rotated, 1))
                rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
            scores[category] = max(
                float(cv2.matchTemplate(
                    variant, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                for variant in variants)
        category = max(scores, key=scores.get)
        return category, scores[category], scores

    def _median_depth(self, depth, quad, center):
        mask = np.zeros(depth.shape, dtype=np.uint8)
        inset = center + 0.72 * (quad - center)
        cv2.fillConvexPoly(mask, np.round(inset).astype(np.int32), 255)
        values = depth[mask > 0]
        values = values[np.isfinite(values)]
        values = values[(values >= self.min_depth) & (values <= self.max_depth)]
        if values.size < 9:
            return None
        return float(np.median(values))

    def _rgb_fallback_pose(self, candidate, header):
        width = 0.5 * (np.linalg.norm(candidate['quad'][1] - candidate['quad'][0])
                        + np.linalg.norm(candidate['quad'][2] - candidate['quad'][3]))
        height = 0.5 * (np.linalg.norm(candidate['quad'][3] - candidate['quad'][0])
                        + np.linalg.norm(candidate['quad'][2] - candidate['quad'][1]))
        pixels = max(min(width, height), 1.0)
        focal = 0.5 * (self.camera_model.fx() + self.camera_model.fy())
        z = focal * self.rgb_cube_size / pixels
        if not self.rgb_pose_min_depth <= z <= self.rgb_pose_max_depth:
            return None
        u, v = candidate['center']
        ray = np.asarray(self.camera_model.projectPixelTo3dRay((float(u), float(v))), dtype=np.float64)
        point = PointStamped()
        point.header = header
        point.point.x = float(ray[0] * z / ray[2])
        point.point.y = float(ray[1] * z / ray[2])
        point.point.z = float(z)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame, header.frame_id, header.stamp,
                rospy.Duration(0.08))
            transformed = tf2_geometry_msgs.do_transform_point(point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, 'cube_vision RGB pose TF unavailable: %s', exc)
            return None
        pose = PoseStamped()
        pose.header = transformed.header
        pose.pose.position = transformed.point
        pose.pose.orientation.w = 1.0
        return pose

    def _candidate_pose(self, candidate, header):
        u, v = candidate['center']
        ray = np.asarray(
            self.camera_model.projectPixelTo3dRay((float(u), float(v))),
            dtype=np.float64)
        if abs(ray[2]) < 1e-6:
            return None
        z = candidate['depth'] + self.surface_to_center_depth
        point = PointStamped()
        point.header = header
        point.point.x = float(ray[0] * z / ray[2])
        point.point.y = float(ray[1] * z / ray[2])
        point.point.z = float(z)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.output_frame, header.frame_id, header.stamp,
                rospy.Duration(0.08))
            transformed = tf2_geometry_msgs.do_transform_point(point, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, 'cube_vision TF unavailable: %s', exc)
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

    def _publish_unknown(self, confidence):
        self.category_pub.publish(String(data='unknown'))
        self.confidence_pub.publish(
            Float32(data=max(0.0, float(confidence))))

    @staticmethod
    def _draw_candidate(image, candidate, valid):
        color = (0, 200, 0) if valid else (0, 165, 255)
        quad = np.round(candidate['quad']).astype(np.int32)
        cv2.polylines(image, [quad], True, color, 2)
        x, y = np.round(candidate['center']).astype(int)
        depth_text = ('{:.2f}m'.format(candidate['depth'])
                      if np.isfinite(candidate['depth']) else 'no-depth')
        label = '{} {:.2f} {}'.format(
            candidate['category'], candidate['score'], depth_text)
        cv2.putText(image, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _publish_debug(self, image, source_msg):
        if not self.publish_debug_image:
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
