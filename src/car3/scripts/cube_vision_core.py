#!/usr/bin/env python3
import collections
import os

import cv2
import numpy as np


CATEGORIES = ('food', 'daily', 'electronics')
TEMPLATE_FILES = {
    'food': 'Food.png',
    'daily': 'Daily_Necessities.png',
    'electronics': 'Electronics.png',
}


class CubeVisionCore:
    def __init__(self, template_dir, params=None):
        params = params or {}
        if not hasattr(cv2, 'SIFT_create'):
            raise RuntimeError('OpenCV SIFT support is required for cube recognition')

        self.brightness_threshold = int(params.get('brightness_threshold', 80))
        self.brightness_offset = int(params.get('brightness_offset', 20))
        self.brightness_blur = int(params.get('brightness_blur', 0))
        self.min_candidate_area = float(params.get('min_candidate_area', 300.0))
        self.max_candidate_area = float(params.get('max_candidate_area', 15000.0))
        self.min_candidate_pixels = int(params.get('min_candidate_pixels', 20))
        self.max_candidate_pixels = int(params.get('max_candidate_pixels', 180))
        self.max_candidate_aspect = float(params.get('max_candidate_aspect', 2.5))
        self.min_candidate_fill = float(params.get('min_candidate_fill', 0.45))
        self.min_candidate_solidity = float(params.get('min_candidate_solidity', 0.82))
        self.min_dark_fraction = float(params.get('min_dark_fraction', 0.04))
        self.dark_pixel_offset = int(params.get('dark_pixel_offset', 18))
        self.roi_padding = int(params.get('roi_padding', 4))
        self.feature_scale = float(params.get('feature_scale', 3.0))
        self.feature_ratio = float(params.get('feature_ratio', 0.74))
        self.feature_ransac_threshold = float(
            params.get('feature_ransac_threshold', 7.0))
        self.min_feature_matches = int(params.get('min_feature_matches', 4))
        self.min_feature_inliers = int(params.get('min_feature_inliers', 4))
        self.min_template_coverage = float(
            params.get('min_template_coverage', 0.008))
        self.min_roi_coverage = float(params.get('min_roi_coverage', 0.004))
        self.min_category_margin = float(params.get('min_category_margin', 0.45))
        self.template_category_fraction = float(
            params.get('template_category_fraction', 0.55))

        self.detector = cv2.SIFT_create(
            nfeatures=700, contrastThreshold=0.01, edgeThreshold=8, sigma=1.2)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.templates = self._load_templates(template_dir)
        self.last_diagnostics = collections.Counter()

    def _load_templates(self, template_dir):
        templates = {}
        for category, filename in TEMPLATE_FILES.items():
            path = os.path.join(template_dir, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError('Cannot load cube template: {}'.format(path))
            category_height = max(
                1, int(round(image.shape[0] * self.template_category_fraction)))
            image = image[:category_height]
            prepared = self._prepare_feature_image(image)
            keypoints, descriptors = self.detector.detectAndCompute(prepared, None)
            if descriptors is None or len(keypoints) < self.min_feature_matches:
                raise RuntimeError('Too few SIFT features in template: {}'.format(path))
            templates[category] = {
                'image': prepared,
                'keypoints': keypoints,
                'descriptors': descriptors,
            }
        return templates

    def _prepare_feature_image(self, image):
        if self.feature_scale != 1.0:
            image = cv2.resize(
                image, None, fx=self.feature_scale, fy=self.feature_scale,
                interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(image)

    def detect(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = (cv2.GaussianBlur(gray, (3, 3), 0)
                   if self.brightness_blur else gray)
        adaptive_floor = int(np.median(blurred)) + self.brightness_offset
        threshold = max(self.brightness_threshold, adaptive_floor)
        bright = np.uint8(blurred >= threshold) * 255
        # Do not dilate this mask: at the search horizon even one added pixel
        # can connect the cube to the broad, bright ground strip. The printed
        # faces remain connected without morphology in the Gazebo render.

        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
        diagnostics = collections.Counter(components=max(0, count - 1))
        candidates = []
        image_height, image_width = gray.shape

        for label in range(1, count):
            x, y, width, height, area = [int(v) for v in stats[label]]
            if not self.min_candidate_area <= area <= self.max_candidate_area:
                diagnostics['area'] += 1
                continue
            short_side = min(width, height)
            long_side = max(width, height)
            if (short_side < self.min_candidate_pixels
                    or long_side > self.max_candidate_pixels
                    or long_side / max(short_side, 1) > self.max_candidate_aspect):
                diagnostics['pixel_size'] += 1
                continue

            component = np.uint8(labels == label) * 255
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            fill = float(area) / max(width * height, 1)
            solidity = contour_area / max(hull_area, 1.0)
            if fill < self.min_candidate_fill or solidity < self.min_candidate_solidity:
                diagnostics['shape'] += 1
                continue

            candidate_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(candidate_mask, hull, 255)
            values = gray[candidate_mask > 0]
            local_bright = float(np.percentile(values, 75))
            dark_fraction = float(np.mean(values < local_bright - self.dark_pixel_offset))
            if dark_fraction < self.min_dark_fraction:
                diagnostics['texture'] += 1
                continue

            x0 = max(0, x - self.roi_padding)
            y0 = max(0, y - self.roi_padding)
            x1 = min(image_width, x + width + self.roi_padding)
            y1 = min(image_height, y + height + self.roi_padding)
            classification = self._classify(gray[y0:y1, x0:x1], (x0, y0))
            center = np.mean(hull.reshape(-1, 2), axis=0).astype(np.float32)
            candidate = {
                'bbox': (x, y, width, height),
                'hull': hull.reshape(-1, 2).astype(np.float32),
                'mask': candidate_mask,
                'center': center,
                'width': float(width),
                'height': float(height),
                'fill': fill,
                'solidity': solidity,
                'dark_fraction': dark_fraction,
                'category': classification['category'],
                'score': classification['confidence'],
                'valid_match': classification['valid'],
                'category_scores': classification['scores'],
                'match_details': classification['details'],
                'face_quad': classification['face_quad'],
                'threshold': threshold,
            }
            candidates.append(candidate)
            diagnostics['candidates'] += 1
            if classification['valid']:
                diagnostics['classified'] += 1

        candidates.sort(
            key=lambda item: (item['valid_match'], item['score'], item['dark_fraction']),
            reverse=True)
        self.last_diagnostics = diagnostics
        return candidates, bright

    @staticmethod
    def _bright_horizon(gray, threshold):
        rows = np.mean(gray >= threshold, axis=1)
        runs = 0
        for index in range(len(rows) - 1, -1, -1):
            if rows[index] > 0.55:
                runs += 1
            elif runs >= 2:
                return index + 1
            else:
                runs = 0
        return len(rows)

    def _classify(self, roi, roi_origin):
        prepared = self._prepare_feature_image(roi)
        roi_keypoints, roi_descriptors = self.detector.detectAndCompute(prepared, None)
        scores = {}
        details = {}
        homographies = {}

        for category, template in self.templates.items():
            evidence = self._match_template(
                template, prepared, roi_keypoints, roi_descriptors)
            scores[category] = evidence['evidence']
            details[category] = evidence
            homographies[category] = evidence['homography']

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        category, best_score = ordered[0]
        margin = best_score - ordered[1][1]
        best = details[category]
        valid = (
            best['matches'] >= self.min_feature_matches
            and best['inliers'] >= self.min_feature_inliers
            and best['template_coverage'] >= self.min_template_coverage
            and best['roi_coverage'] >= self.min_roi_coverage
            and margin >= self.min_category_margin)

        confidence = self._confidence(best, margin) if valid else 0.0
        face_quad = None
        homography = homographies[category]
        if valid and homography is not None:
            template_image = self.templates[category]['image']
            corners = np.float32([[
                [0, 0], [template_image.shape[1] - 1, 0],
                [template_image.shape[1] - 1, template_image.shape[0] - 1],
                [0, template_image.shape[0] - 1],
            ]])
            projected = cv2.perspectiveTransform(corners, homography)[0]
            projected /= self.feature_scale
            if self._valid_face_quad(projected, roi.shape, self.feature_scale):
                projected[:, 0] += roi_origin[0]
                projected[:, 1] += roi_origin[1]
                face_quad = projected.astype(np.float32)

        return {
            'category': category,
            'confidence': confidence,
            'valid': valid,
            'scores': scores,
            'details': details,
            'face_quad': face_quad,
        }

    def _match_template(self, template, roi, roi_keypoints, roi_descriptors):
        empty = {
            'matches': 0, 'inliers': 0, 'template_coverage': 0.0,
            'roi_coverage': 0.0, 'evidence': 0.0, 'homography': None,
        }
        if roi_descriptors is None or len(roi_keypoints) < 2:
            return empty

        pairs = self.matcher.knnMatch(template['descriptors'], roi_descriptors, k=2)
        good = [first for first, second in pairs
                if first.distance < self.feature_ratio * second.distance]
        unique = []
        used_template = set()
        used_roi = set()
        for match in sorted(good, key=lambda item: item.distance):
            if match.queryIdx in used_template or match.trainIdx in used_roi:
                continue
            used_template.add(match.queryIdx)
            used_roi.add(match.trainIdx)
            unique.append(match)

        result = dict(empty)
        result['matches'] = len(unique)
        if len(unique) < self.min_feature_matches:
            result['evidence'] = 0.01 * len(unique)
            return result

        source = np.float32([
            template['keypoints'][match.queryIdx].pt for match in unique])
        destination = np.float32([
            roi_keypoints[match.trainIdx].pt for match in unique])
        homography, inlier_mask = cv2.findHomography(
            source, destination, cv2.RANSAC, self.feature_ransac_threshold)
        if homography is None or inlier_mask is None:
            result['evidence'] = 0.01 * len(unique)
            return result

        inliers = inlier_mask.reshape(-1) > 0
        source_inliers = source[inliers]
        destination_inliers = destination[inliers]
        inlier_count = int(np.count_nonzero(inliers))
        template_coverage = self._point_coverage(
            source_inliers, template['image'].shape)
        roi_coverage = self._point_coverage(destination_inliers, roi.shape)
        result.update({
            'inliers': inlier_count,
            'template_coverage': template_coverage,
            'roi_coverage': roi_coverage,
            'homography': homography,
            'evidence': (
                inlier_count
                + 0.5 * min(template_coverage / 0.05, 1.0)
                + 0.5 * min(roi_coverage / 0.01, 1.0)
                + 0.01 * len(unique)),
        })
        return result

    @staticmethod
    def _valid_face_quad(quad, scaled_roi_shape, feature_scale):
        if not np.all(np.isfinite(quad)) or not cv2.isContourConvex(quad.astype(np.float32)):
            return False
        roi_height = scaled_roi_shape[0] / feature_scale
        roi_width = scaled_roi_shape[1] / feature_scale
        area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        if not 0.01 * roi_width * roi_height <= area <= 1.5 * roi_width * roi_height:
            return False
        lengths = [np.linalg.norm(quad[(index + 1) % 4] - quad[index])
                   for index in range(4)]
        return min(lengths) >= 3.0 and max(lengths) / min(lengths) <= 6.0

    @staticmethod
    def _point_coverage(points, image_shape):
        if len(points) < 3:
            return 0.0
        area = float(cv2.contourArea(cv2.convexHull(points)))
        return area / max(float(image_shape[0] * image_shape[1]), 1.0)

    def _confidence(self, evidence, margin):
        inlier_term = min(
            1.0, evidence['inliers'] / float(self.min_feature_inliers + 2))
        coverage_term = min(
            1.0, evidence['roi_coverage'] / max(self.min_roi_coverage * 3.0, 1e-6))
        margin_term = min(1.0, margin / max(self.min_category_margin * 2.0, 1e-6))
        return float(np.clip(
            0.45 + 0.25 * inlier_term + 0.15 * coverage_term
            + 0.15 * margin_term, 0.0, 1.0))

    @staticmethod
    def inset_mask(candidate, scale=0.82):
        polygon = (candidate['face_quad']
                   if candidate.get('face_quad') is not None
                   else candidate['hull'])
        center = np.mean(polygon, axis=0)
        inset = center + float(scale) * (polygon - center)
        mask = np.zeros(candidate['mask'].shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(inset).astype(np.int32), 255)
        return mask

    @staticmethod
    def estimate_cube_center(depth, mask, camera_matrix, half_size,
                             min_depth, max_depth, min_points=30,
                             max_residual=0.004):
        depth = np.asarray(depth)
        # Some Gazebo/CvBridge combinations expose a single-channel depth
        # image as HxWx1. Normalize it before advanced pixel indexing; a
        # multi-channel depth image is not valid input for this geometry.
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2 or mask.ndim != 2 or depth.shape != mask.shape:
            return None
        rows, cols = np.nonzero(mask > 0)
        values = depth[rows, cols].astype(np.float64)
        valid = (np.isfinite(values)
                 & (values >= float(min_depth))
                 & (values <= float(max_depth)))
        rows = rows[valid]
        cols = cols[valid]
        values = values[valid]
        if values.size < int(min_points):
            return None

        median = float(np.median(values))
        depth_band = max(0.015, 4.0 * float(max_residual))
        near_surface = np.abs(values - median) <= depth_band
        rows = rows[near_surface]
        cols = cols[near_surface]
        values = values[near_surface]
        if values.size < int(min_points):
            return None

        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        points = np.column_stack((
            (cols.astype(np.float64) - cx) * values / fx,
            (rows.astype(np.float64) - cy) * values / fy,
            values,
        ))

        inliers = np.ones(points.shape[0], dtype=bool)
        normal = None
        surface_origin = None
        for _ in range(3):
            selected = points[inliers]
            if selected.shape[0] < int(min_points):
                return None
            surface_origin = np.median(selected, axis=0)
            _, _, axes = np.linalg.svd(selected - surface_origin,
                                       full_matrices=False)
            normal = axes[-1]
            residuals = np.abs((points - surface_origin).dot(normal))
            next_inliers = residuals <= float(max_residual)
            if np.array_equal(next_inliers, inliers):
                break
            inliers = next_inliers

        if normal is None or np.count_nonzero(inliers) < int(min_points):
            return None
        selected = points[inliers]
        surface_origin = np.mean(selected, axis=0)
        residual_rms = float(np.sqrt(np.mean(
            np.square((selected - surface_origin).dot(normal)))))
        if residual_rms > float(max_residual):
            return None

        ray = np.array([
            (float(np.mean(cols[inliers])) - cx) / fx,
            (float(np.mean(rows[inliers])) - cy) / fy,
            1.0,
        ], dtype=np.float64)
        denominator = float(np.dot(normal, ray))
        if abs(denominator) < 0.15:
            return None
        distance = float(np.dot(normal, surface_origin) / denominator)
        if distance <= 0.0:
            return None
        surface = distance * ray

        # The outward normal of a visible face points toward the camera.
        if np.dot(normal, -surface) < 0.0:
            normal = -normal
        center = surface - float(half_size) * normal
        return {
            'center': center,
            'surface': surface,
            'normal': normal,
            'depth': float(surface[2]),
            'valid_points': int(np.count_nonzero(inliers)),
            'residual_rms': residual_rms,
        }
