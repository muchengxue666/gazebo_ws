#!/usr/bin/env python3
import math

import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Point, Pose, Quaternion


# These bounds mirror spawn_cubes.py. The marker models are visual-only and
# have no collision geometry, so they do not affect navigation or laser scans.
CUBE_AREAS = [
    ('area_a', -2.10, -1.92, -0.61, -0.28, (0.95, 0.15, 0.15, 0.75)),
    ('area_b', -1.56, -1.23, -0.01, 0.17, (0.15, 0.85, 0.20, 0.75)),
    ('area_c', -0.95, -0.77, -0.69, -0.36, (0.15, 0.35, 0.95, 0.75)),
]

MARKER_PREFIX = 'cube_area_marker_'
EDGE_THICKNESS = 0.018
EDGE_HEIGHT = 0.012
CENTER_RADIUS = 0.035


def _material(color):
    r, g, b, a = color
    return '''
      <material>
        <ambient>{r} {g} {b} 1</ambient>
        <diffuse>{r} {g} {b} 1</diffuse>
        <emissive>{r} {g} {b} 0.15</emissive>
        <transparency>{transparency}</transparency>
      </material>'''.format(r=r, g=g, b=b, transparency=1.0 - a)


def _box_visual(name, x, y, sx, sy, color):
    return '''
      <visual name="{name}">
        <pose>{x} {y} 0 0 0 0</pose>
        <geometry>
          <box><size>{sx} {sy} {height}</size></box>
        </geometry>
        {material}
      </visual>'''.format(
        name=name, x=x, y=y, sx=sx, sy=sy, height=EDGE_HEIGHT,
        material=_material(color))


def _center_visual(color):
    return '''
      <visual name="center_marker">
        <pose>0 0 0.002 0 0 0</pose>
        <geometry>
          <cylinder><radius>{radius}</radius><length>{height}</length></cylinder>
        </geometry>
        {material}
      </visual>'''.format(
        radius=CENTER_RADIUS, height=EDGE_HEIGHT,
        material=_material(color))


def _marker_sdf(area_name, x_min, x_max, y_min, y_max, color):
    width = x_max - x_min
    height = y_max - y_min
    half_width = 0.5 * width
    half_height = 0.5 * height
    visuals = [
        _box_visual('left_edge', -half_width, 0.0, EDGE_THICKNESS,
                    height, color),
        _box_visual('right_edge', half_width, 0.0, EDGE_THICKNESS,
                    height, color),
        _box_visual('bottom_edge', 0.0, -half_height, width,
                    EDGE_THICKNESS, color),
        _box_visual('top_edge', 0.0, half_height, width,
                    EDGE_THICKNESS, color),
        _center_visual(color),
    ]
    return '''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      {visuals}
    </link>
  </model>
</sdf>
'''.format(
        model_name=MARKER_PREFIX + area_name,
        visuals='\n'.join(visuals))


def _delete_existing(delete_model):
    for area_name, *_ in CUBE_AREAS:
        try:
            delete_model(MARKER_PREFIX + area_name)
        except rospy.ServiceException:
            pass


def main():
    rospy.init_node('mark_cube_areas')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    rospy.wait_for_service('/gazebo/delete_model')
    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    _delete_existing(delete_model)

    for area_name, x_min, x_max, y_min, y_max, color in CUBE_AREAS:
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        pose = Pose(
            position=Point(x=center_x, y=center_y, z=0.006),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
        model_name = MARKER_PREFIX + area_name
        response = spawn_model(
            model_name,
            _marker_sdf(area_name, x_min, x_max, y_min, y_max, color),
            '', pose, 'world')
        if not response.success:
            raise RuntimeError('{}: {}'.format(model_name, response.status_message))
        rospy.loginfo(
            '%s: x=[%.2f, %.2f], y=[%.2f, %.2f]',
            area_name, x_min, x_max, y_min, y_max)

    rospy.loginfo('cube generation areas marked in Gazebo')
    rospy.spin()


if __name__ == '__main__':
    main()
