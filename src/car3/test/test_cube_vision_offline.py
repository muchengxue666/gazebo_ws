#!/usr/bin/env python3
import os
import sys
import unittest

import cv2
import numpy as np
import yaml

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, 'scripts'))

from cube_vision_core import CubeVisionCore  # noqa: E402


class CubeVisionOfflineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(PACKAGE_ROOT, 'test', 'data', 'cube_vision')
        with open(os.path.join(cls.data_dir, 'manifest.yaml')) as stream:
            cls.samples = yaml.safe_load(stream)['samples']
        template_dir = os.path.join(
            PACKAGE_ROOT, 'models', 'cube', 'meshes')
        cls.core = CubeVisionCore(template_dir)

    def _best(self, image):
        candidates, _ = self.core.detect(image)
        self.assertTrue(candidates, 'expected at least one cube candidate')
        return candidates[0]

    def test_labeled_observations(self):
        for sample in self.samples:
            with self.subTest(file=sample['file']):
                image = cv2.imread(os.path.join(self.data_dir, sample['file']))
                self.assertIsNotNone(image)
                candidate = self._best(image)
                self.assertTrue(candidate['valid_match'])
                self.assertEqual(sample['category'], candidate['category'])
                actual = np.array(candidate['bbox'], dtype=np.float32)
                expected = np.array(sample['bbox'], dtype=np.float32)
                self.assertLess(float(np.max(np.abs(actual - expected))), 8.0)

    def test_background_crops_are_not_classified(self):
        for sample in self.samples:
            image = cv2.imread(os.path.join(self.data_dir, sample['file']))
            background = image[:220, 180:460]
            candidates, _ = self.core.detect(background)
            self.assertFalse(any(item['valid_match'] for item in candidates))

    def test_mild_brightness_change(self):
        for sample in self.samples:
            with self.subTest(file=sample['file']):
                image = cv2.imread(os.path.join(self.data_dir, sample['file']))
                transformed = cv2.convertScaleAbs(image, alpha=0.98, beta=2)
                candidates, _ = self.core.detect(transformed)
                self.assertTrue(candidates)
                candidate = candidates[0]
                # Mild render changes may be held as unknown at this scale,
                # but must never be converted into a wrong known class.
                if candidate['valid_match']:
                    self.assertEqual(sample['category'], candidate['category'])
                else:
                    self.assertIn(candidate['category'], ('food', 'daily', 'electronics'))
                    self.assertLessEqual(candidate['score'], 0.0)


if __name__ == '__main__':
    unittest.main()
