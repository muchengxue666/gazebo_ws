#!/usr/bin/env python3
import math
import os
import sys
import types
import unittest
from unittest import mock

import yaml
from geometry_msgs.msg import PoseStamped

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PACKAGE_ROOT, 'scripts'))

from pick_place_executor import PickPlaceExecutor  # noqa: E402


def _pose(x, y):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.w = 1.0
    return pose


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


class PickPlaceSearchTest(unittest.TestCase):
    def _executor(self, fresh_pose):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        executor.target_category = 'daily'
        executor.search_settle_time = 0.5
        executor.search_target_confirm_timeout = 4.0
        executor.search_target_confirm_distance = 0.05
        executor.vision_pose_seq = 7
        executor.vision_pose_confidence = 0.8
        executor.detected_category_pub = _Publisher()
        executor._stop_base = mock.Mock()
        executor._set_state = mock.Mock()
        executor.vision_reset = mock.Mock()
        executor._fresh_target_pose = mock.Mock(return_value=fresh_pose)
        executor._point_in_frame = lambda pose, _frame: types.SimpleNamespace(
            point=pose.pose.position)
        return executor

    @mock.patch('pick_place_executor.time.sleep')
    def test_stopped_confirmation_accepts_nearby_fresh_pose(self, _sleep):
        fresh_pose = _pose(-1.40, 0.02)
        executor = self._executor(fresh_pose)

        result = executor._confirm_stopped_search_target(
            _pose(-1.42, 0.01), 'coarse search')

        self.assertIs(result, fresh_pose)
        executor._stop_base.assert_called_once_with()
        executor._set_state.assert_called_once_with('CONFIRM_CUBE')
        executor.vision_reset.assert_called_once_with()
        executor._fresh_target_pose.assert_called_once_with(7, timeout=4.0)
        self.assertEqual(['daily'], executor.detected_category_pub.messages)

    @mock.patch('pick_place_executor.rospy.is_shutdown', return_value=False)
    def test_task_start_is_prepared_once_before_category(self, _shutdown):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        executor.startup_prepared = False
        executor.arm_poses_verified = True
        executor.nav_client = object()
        executor.arm_client = object()
        executor.gripper_open = 1.5
        executor.grasp_state = 'IDLE'
        executor.attached_model = ''
        executor._set_state = mock.Mock()
        executor._wait_for_action_server = mock.Mock(return_value=True)
        executor._wait_joint_state = mock.Mock(return_value=True)
        executor._command_gripper = mock.Mock()
        executor._move_arm = mock.Mock()
        executor._wait_for_navigation_ready = mock.Mock()
        executor._log_timing = mock.Mock()

        executor._prepare_task_start()
        executor._prepare_task_start()

        self.assertTrue(executor.startup_prepared)
        self.assertEqual(2, executor._wait_for_action_server.call_count)
        executor._command_gripper.assert_called_once_with(1.5, 'open')
        executor._move_arm.assert_called_once_with('navigation')
        executor._wait_for_navigation_ready.assert_called_once_with()

    @mock.patch('pick_place_executor.time.sleep')
    def test_stopped_confirmation_rejects_shifted_pose(self, _sleep):
        executor = self._executor(_pose(-1.30, 0.10))

        result = executor._confirm_stopped_search_target(
            _pose(-1.42, 0.01), 'coarse search')

        self.assertIsNone(result)
        self.assertEqual([], executor.detected_category_pub.messages)

    @mock.patch('pick_place_executor.time.sleep')
    def test_coarse_center_rotates_edge_candidate_toward_camera_center(self, _sleep):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        executor.coarse_center_enabled = True
        executor.coarse_center_camera_frame = 'camera_depth_optical_frame'
        executor.coarse_center_tolerance = 0.08
        executor.coarse_center_max_angle = 0.65
        executor.coarse_center_timeout = 1.0
        executor.coarse_center_gain = 1.5
        executor.coarse_center_max_speed = 0.30
        executor.search_rotation_rate = 20.0
        executor.cmd_vel_pub = mock.Mock()
        executor._stop_base = mock.Mock()
        executor._set_state = mock.Mock()
        candidate = _pose(-1.4, -0.4)
        points = [types.SimpleNamespace(x=0.25, z=-1.0),
                  types.SimpleNamespace(x=0.25, z=-1.0),
                  types.SimpleNamespace(x=0.04, z=-1.0)]
        executor._point_in_current_frame = mock.Mock(
            side_effect=lambda _pose, _frame: types.SimpleNamespace(
                point=points.pop(0)))

        centered = executor._center_coarse_view(candidate, 'coarse search')

        self.assertTrue(centered)
        command = executor.cmd_vel_pub.publish.call_args.args[0]
        self.assertGreater(command.angular.z, 0.0)
        executor._stop_base.assert_called_once_with()

    def test_grasp_preparation_approaches_above_before_descending(self):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        executor.ready = True
        executor.attach_offset = (0.0, 0.0, 0.0)
        executor.max_grasp_offset = 0.02
        executor._set_state = mock.Mock()
        executor._move_arm = mock.Mock()
        executor._log_timing = mock.Mock()

        executor._prepare_stationary_grasp()

        self.assertEqual(
            [mock.call('grasp_approach'), mock.call('grasp')],
            executor._move_arm.call_args_list)

    def test_attached_cube_is_lifted_before_transport(self):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        executor.ready = True
        executor.gripper_close = 0.8
        executor.grasp_state = 'GRASPING'
        executor.attached_model = 'cube_1'
        executor.target_category = 'daily'
        executor.category_to_model = {'daily': 'cube_1'}
        events = []
        executor._set_state = lambda state: events.append(('state', state))
        executor._command_gripper = mock.Mock()
        executor._wait_attach = lambda: events.append(('attached', None))
        executor._wait_attached_gripper_settle = mock.Mock()
        executor._move_arm = lambda name: events.append(('arm', name))
        executor._check_attachment = lambda: events.append(('checked', None))
        executor._log_timing = mock.Mock()

        executor._grasp()

        self.assertLess(
            events.index(('attached', None)),
            events.index(('arm', 'grasp_lift')))
        self.assertLess(
            events.index(('arm', 'grasp_lift')),
            events.index(('checked', None)))

    def test_lift_reverses_the_verified_grasp_approach(self):
        with open(os.path.join(
                PACKAGE_ROOT, 'config', 'nav_pick_place_task.yaml'),
                encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
        approach = config['arm_poses']['grasp_approach']
        lift = config['arm_poses']['grasp_lift']

        # Grasp approach was executed collision-free before the gripper closed;
        # returning through its exact reverse avoids introducing an untested
        # self-folding lift trajectory after an object is attached.
        self.assertNotEqual(
            config['arm_poses']['grasp']['positions'], lift['positions'])
        self.assertEqual(approach['positions'], lift['positions'])
        self.assertEqual(approach['duration'], lift['duration'])

    def test_search_visits_all_nominal_areas_before_standoff_retry(self):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        executor.fast_area_search = False
        executor.coarse_search_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        executor.area_search_passes = 2
        executor.area_search_pan_offsets = [0.45, -0.45]
        executor.area_search_waypoints = [
            {'area': 'area_a'}, {'area': 'area_b'}, {'area': 'area_c'}]
        executor.area_search_view_stages = [
            {'pose': 'observe', 'pan_offsets': []},
            {'pose': 'area_observe', 'pan_offsets': [0.45, -0.45]}]
        executor._set_state = mock.Mock()
        executor._log_timing = mock.Mock()
        executor._rotate_search_at_pose = mock.Mock(return_value=None)
        executor._search_area_waypoint = mock.Mock(return_value=None)

        with self.assertRaisesRegex(RuntimeError, 'multi-view area searches'):
            executor._search()

        self.assertEqual(
            [
                mock.call(executor.area_search_waypoints[0], 0,
                          _allow_standoff=False),
                mock.call(executor.area_search_waypoints[1], 1,
                          _allow_standoff=False),
                mock.call(executor.area_search_waypoints[2], 2,
                          _allow_standoff=False),
                mock.call(executor.area_search_waypoints[0], 0,
                          _allow_standoff=True),
                mock.call(executor.area_search_waypoints[1], 1,
                          _allow_standoff=True),
                mock.call(executor.area_search_waypoints[2], 2,
                          _allow_standoff=True),
            ],
            executor._search_area_waypoint.call_args_list)

    def test_confirmed_coarse_target_enters_its_fixed_area_search(self):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        coarse_pose = _pose(-1.98, -0.45)
        refined_pose = _pose(-1.97, -0.45)
        executor.fast_area_search = False
        executor.coarse_search_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        executor.area_search_waypoints = [
            {'area': 'area_a'}, {'area': 'area_b'}, {'area': 'area_c'}]
        executor.area_search_passes = 1
        executor._set_state = mock.Mock()
        executor._log_timing = mock.Mock()
        executor._rotate_search_at_pose = mock.Mock(return_value=coarse_pose)
        executor._search_area_name = mock.Mock(return_value='area_a')
        executor._search_area_waypoint = mock.Mock(return_value=refined_pose)

        result = executor._search()

        self.assertIs(result, refined_pose)
        executor._search_area_waypoint.assert_called_once_with(
            executor.area_search_waypoints[0], 0, _allow_standoff=False)

    @mock.patch('pick_place_executor.time.sleep')
    def test_coarse_search_stops_and_confirms_non_target(self, _sleep):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        executor.target_category = 'electronics'
        executor.coarse_search_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        executor.search_settle_time = 0.0
        executor.search_rotation_speed = 0.2
        executor.search_rotation_angle = 0.01
        executor.search_rotation_timeout = 5.0
        executor.search_rotation_rate = 20.0
        executor.search_rotation_max_drift = 0.08
        executor.vision_pose_received = 0.0
        executor.cmd_vel_pub = mock.Mock()
        executor._stop_base = mock.Mock()
        executor._move_arm = mock.Mock()
        executor._set_state = mock.Mock()
        executor._log_timing = mock.Mock()
        executor._base_pose_odom = mock.Mock(
            side_effect=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                         (0.0, 0.0, 0.0), (0.0, 0.0, 0.02),
                         (0.0, 0.0, 0.04), (0.0, 0.0, 0.04)])
        executor._current_detection = mock.Mock(return_value=None)
        executor._current_known_detection = mock.Mock(
            side_effect=[None, ('food', _pose(-1.5, -0.4)),
                         ('food', _pose(-1.5, -0.4)), None])
        executor._confirm_search_target = mock.Mock(
            return_value=(True, ('food', _pose(-1.5, -0.4))))
        executor._is_seen_non_target = mock.Mock(
            side_effect=[False, False, True])
        executor._record_non_target = mock.Mock()

        result = executor._rotate_search_at_pose(
            executor.coarse_search_pose, navigate_to_pose=False)

        self.assertIsNone(result)
        executor._confirm_search_target.assert_called_once_with(
            mock.ANY, 'coarse search', center_view=True)
        executor._record_non_target.assert_called_once()
        # A confirmed non-target remains inside coarse rotation; fixed-area
        # search is only entered after the full coarse sweep or target hit.
        self.assertEqual([mock.call('observe'), mock.call('navigation')],
                         executor._move_arm.call_args_list)

    def test_area_waypoints_are_close_and_use_a_distinct_arm_pose(self):
        config_path = os.path.join(
            PACKAGE_ROOT, 'config', 'nav_pick_place_task.yaml')
        with open(config_path, encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)

        self.assertNotEqual(
            config['arm_poses']['observe']['positions'],
            config['arm_poses']['area_observe']['positions'])
        self.assertEqual(
            ['observe', 'area_observe'],
            [stage['pose'] for stage in config['area_search_view_stages']])
        self.assertEqual([], config['area_search_view_stages'][0]['pan_offsets'])
        self.assertEqual(
            [0.45, -0.45],
            config['area_search_view_stages'][1]['pan_offsets'])
        for waypoint in config['area_search_waypoints']:
            bounds = config['search_areas'][waypoint['area']]
            center_x = 0.5 * (bounds['x_min'] + bounds['x_max'])
            center_y = 0.5 * (bounds['y_min'] + bounds['y_max'])
            distance = ((waypoint['x'] - center_x) ** 2
                        + (waypoint['y'] - center_y) ** 2) ** 0.5
            self.assertAlmostEqual(0.45, distance, delta=0.01)
            expected_yaw = math.atan2(
                center_y - waypoint['y'], center_x - waypoint['x'])
            yaw_delta = math.atan2(
                math.sin(waypoint['yaw'] - expected_yaw),
                math.cos(waypoint['yaw'] - expected_yaw))
            self.assertAlmostEqual(0.0, yaw_delta, delta=0.01)

        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        executor.search_areas = config['search_areas']
        for waypoint in config['area_search_waypoints']:
            self.assertAlmostEqual(
                waypoint['yaw'],
                executor._area_observation_yaw(waypoint['area'], waypoint),
                delta=0.01)

    def test_area_view_joint1_offset_uses_absolute_camera_heading(self):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)

        # With the base already aligned to the area, the camera must point
        # forward (joint1 ~= 0), not retain the old nominal 90 degree pan.
        offset = executor._area_view_joint1_offset(
            nominal_joint1=1.570006,
            target_yaw=1.20,
            actual_yaw=1.20)
        self.assertAlmostEqual(-1.570006, offset, places=5)
        self.assertAlmostEqual(
            0.0, 1.570006 + offset, places=5)

        # A nonzero base yaw still produces the absolute relative heading.
        offset = executor._area_view_joint1_offset(
            nominal_joint1=1.570006,
            target_yaw=1.20,
            actual_yaw=0.95)
        self.assertAlmostEqual(
            0.25 - 1.570006, offset, places=5)

    def test_final_grasp_alignment_does_not_reload_nominal_observe_pose(self):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        target_pose = _pose(0.30, 0.0)
        executor.max_align_iterations = 2
        executor.fine_align_distance = 0.08
        executor.grasp_tcp_in_base = (0.289, 0.0, 0.016)
        executor._set_state = mock.Mock()
        executor._classify_search_area = mock.Mock()
        executor._point_in_frame = lambda pose, _frame: types.SimpleNamespace(
            point=pose.pose.position)
        executor._base_pose_map = mock.Mock(return_value=(0.0, 0.0, 0.0))
        executor._move_arm = mock.Mock()
        executor._fine_align_to_grasp = mock.Mock()
        executor._reset_vision_and_observe = mock.Mock(
            side_effect=AssertionError(
                'final alignment must not reload nominal observe'))

        executor._align_to_grasp_with_fine_tolerance(target_pose)

        executor._fine_align_to_grasp.assert_called_once_with(
            target_pose.pose.position, rotate=True)
        executor._reset_vision_and_observe.assert_not_called()

    def test_intermediate_grasp_alignment_keeps_navigation_arm_pose(self):
        executor = PickPlaceExecutor.__new__(PickPlaceExecutor)
        target_pose = _pose(0.50, 0.0)
        executor.max_align_iterations = 2
        executor.max_align_step = 1.0
        executor.fine_align_distance = 0.08
        executor.grasp_tcp_in_base = (0.289, 0.0, 0.016)
        executor._set_state = mock.Mock()
        executor._classify_search_area = mock.Mock()
        executor._point_in_frame = lambda pose, _frame: types.SimpleNamespace(
            point=pose.pose.position)
        executor._base_pose_map = mock.Mock(
            side_effect=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                         (0.20, 0.0, 0.0)])
        executor._navigate = mock.Mock(return_value=True)
        executor._stop_base = mock.Mock()
        executor._move_arm = mock.Mock()
        executor._fine_align_to_grasp = mock.Mock()
        executor._reset_vision_and_observe = mock.Mock(
            side_effect=AssertionError(
                'intermediate alignment must not reload nominal observe'))

        executor._align_to_grasp_with_fine_tolerance(target_pose)

        executor._navigate.assert_called_once()
        executor._fine_align_to_grasp.assert_called_once_with(
            target_pose.pose.position, rotate=True)
        executor._reset_vision_and_observe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
