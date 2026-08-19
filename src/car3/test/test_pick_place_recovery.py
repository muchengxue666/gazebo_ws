#!/usr/bin/env python3
import os
import sys
import unittest
from unittest import mock

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, 'scripts'))

import pick_place_executor as executor_module  # noqa: E402


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


class _Executor:
    instance = None

    def __init__(self):
        type(self).instance = self
        self.target_category = 'food'
        self.target_was_located = True
        self.failure_search_retry_delay = 0.0
        self.result_pub = _Publisher()
        self.states = []
        self.recovery_calls = 0
        self.stop_calls = 0

    def wait_for_target_category(self):
        return None

    def run(self, initial_detection=None, failure_local_detection=False):
        raise RuntimeError('simulated transport arm failure')

    def _base_pose_map(self):
        return 0.0, 0.0, 0.0

    def _recover_after_failure(self):
        self.recovery_calls += 1
        return False

    def _set_state(self, state):
        self.states.append(state)

    def _stop_base(self):
        self.stop_calls += 1


class PickPlaceRecoveryTest(unittest.TestCase):
    def test_grasp_failure_restarts_complete_coarse_search(self):
        class _GraspFailureExecutor:
            instance = None

            def __init__(self):
                type(self).instance = self
                self.target_category = 'electronics'
                self.target_was_located = True
                self.grasp_completed = False
                self.failure_search_retry_delay = 0.0
                self.result_pub = _Publisher()
                self.run_calls = []
                self.recovery_calls = 0
                self.local_search_calls = 0

            def wait_for_target_category(self):
                return None

            def run(self, initial_detection=None, failure_local_detection=False):
                self.run_calls.append(
                    (initial_detection, failure_local_detection))
                if len(self.run_calls) == 1:
                    raise RuntimeError('simulated attachment failure')

            def _base_pose_map(self):
                return 0.0, 0.0, 0.0

            def _recover_after_failure(self):
                self.recovery_calls += 1
                return True

            def _search_near_failure(self, _failure_pose):
                self.local_search_calls += 1
                raise AssertionError('grasp failure must restart coarse search')

            def _set_state(self, _state):
                return None

            def _stop_base(self):
                return None

        with mock.patch.object(
                executor_module, 'PickPlaceExecutor', _GraspFailureExecutor), \
                mock.patch.object(executor_module.rospy, 'init_node'), \
                mock.patch.object(
                    executor_module.rospy, 'is_shutdown', return_value=False), \
                mock.patch.object(executor_module.rospy, 'set_param'), \
                mock.patch.object(executor_module.time, 'sleep'):
            executor_module.main()

        executor = _GraspFailureExecutor.instance
        self.assertEqual(1, executor.recovery_calls)
        self.assertEqual(0, executor.local_search_calls)
        self.assertEqual([(None, False), (None, False)], executor.run_calls)

    def test_unrecoverable_arm_failure_stops_after_bounded_attempts(self):
        with mock.patch.object(
                executor_module, 'PickPlaceExecutor', _Executor), \
                mock.patch.object(executor_module.rospy, 'init_node'), \
                mock.patch.object(
                    executor_module.rospy, 'is_shutdown', return_value=False), \
                mock.patch.object(executor_module.rospy, 'set_param'), \
                mock.patch.object(executor_module.time, 'sleep'):
            executor_module.main()

        executor = _Executor.instance
        self.assertEqual(
            executor_module.MAX_FAILURE_RECOVERY_ATTEMPTS,
            executor.recovery_calls)
        self.assertEqual('FAILED', executor.states[-1])
        self.assertEqual(1, executor.stop_calls)
        self.assertEqual(
            'FAILED:failure recovery did not complete after 3 attempts',
            executor.result_pub.messages[-1])


if __name__ == '__main__':
    unittest.main()
