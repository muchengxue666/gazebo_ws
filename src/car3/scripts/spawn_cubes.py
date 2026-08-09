#!/usr/bin/env python3
import math
import os
import random
import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion

CUBE_AREAS = [
    (-2.10, -1.92, -0.61, -0.28,  1.5708),
    (-1.56, -1.23, -0.01,  0.17,  0.0),
    (-0.95, -0.77, -0.69, -0.36, -1.5708),
]

CONE_COUNT = 10
CONE_SPAWN_X = (-1.9, 2.7)
CONE_SPAWN_Y = (-3.22, -1.26)
CONE_MIN_DIST = 0.62  
BOUNDARY_CLEARANCE = 0.2 

GAP = 0.2
FORBIDDEN = [
    (2.30-GAP, 2.80+GAP, -2.47-GAP, -1.97+GAP), 
    (0.75-GAP, 1.25+GAP, -1.75-GAP, -1.25+GAP), 
    (0.75-GAP, 1.25+GAP, -3.23-GAP, -2.73+GAP), 
    (-1.25-GAP, -0.5+GAP, -1.5-GAP, -1.2+GAP), 
]

CONE_SDF = '''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}"><static>0</static>
    <link name="link">
      <collision name="collision">
        <geometry><mesh><scale>3.976 3.976 3.489</scale>
          <uri>model://construction_cone/meshes/construction_cone.dae</uri></mesh></geometry>
        <surface><contact><ode/></contact><bounce/>
          <friction><torsional><ode/></torsional><ode/></friction></surface>
        <max_contacts>10</max_contacts>
      </collision>
      <visual name="visual">
        <geometry><mesh><scale>3.976 3.976 3.489</scale>
          <uri>model://construction_cone/meshes/construction_cone.dae</uri></mesh></geometry>
      </visual>
      <inertial><pose>0 0 0 0 0 0</pose>
        <inertia><ixx>1</ixx><ixy>0</ixy><ixz>0</ixz><iyy>1</iyy><iyz>0</iyz><izz>1</izz></inertia>
        <mass>1</mass></inertial>
    </link>
  </model>
</sdf>'''

SDF_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'cube')
Z = 0.02


def _in_zone(x, y, zones):
    for x0, x1, y0, y1 in zones:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def _dist_to_others(px, py, placed):
    return min((math.hypot(px - cx, py - cy) for cx, cy in placed), default=float('inf'))


def spawn_cones(spawn_srv):
    eff_x_min = CONE_SPAWN_X[0] + BOUNDARY_CLEARANCE
    eff_x_max = CONE_SPAWN_X[1] - BOUNDARY_CLEARANCE
    eff_y_min = CONE_SPAWN_Y[0] + BOUNDARY_CLEARANCE
    eff_y_max = CONE_SPAWN_Y[1] - BOUNDARY_CLEARANCE

    placed = []
    attempts = 0
    max_attempts = 5000

    while len(placed) < CONE_COUNT and attempts < max_attempts:
        x = random.uniform(eff_x_min, eff_x_max)
        y = random.uniform(eff_y_min, eff_y_max)
        attempts += 1

        if _in_zone(x, y, FORBIDDEN):
            continue
        if _dist_to_others(x, y, placed) < CONE_MIN_DIST:
            continue

        model_name = f'cone_{len(placed) + 10}'
        sdf = CONE_SDF.format(name=model_name)
        pose = Pose(position=Point(x=x, y=y, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1))
        spawn_srv(model_name, sdf, '', pose, 'world')
        rospy.loginfo(f'  {model_name} → ({x:.3f}, {y:.3f})')
        placed.append((x, y))

    if len(placed) < CONE_COUNT:
        rospy.logwarn(f'锥桶: 只生成 {len(placed)}/{CONE_COUNT} (尝试 {attempts} 次)')
    else:
        rospy.loginfo(f'锥桶: {len(placed)}/{CONE_COUNT} 个')


def main():
    rospy.init_node('spawn_cubes')
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    rospy.wait_for_service('/gazebo/delete_model')
    spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    delete_srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

    to_delete = [f'cube_{i}' for i in range(3)]
    to_delete += [f'cone_{i}' for i in range(10, 10 + CONE_COUNT)]
    for name in to_delete:
        try:
            delete_srv(name)
        except rospy.ServiceException:
            pass

    models = {}
    for i in range(3):
        name = f'cube_{i}'
        path = os.path.join(SDF_DIR, f'model_{i}.sdf')
        with open(path) as f:
            models[name] = f.read()

    shuffled = random.sample(CUBE_AREAS, 3)
    cube_names = random.sample(list(models.keys()), 3)
    for area, cname in zip(shuffled, cube_names):
        x_min, x_max, y_min, y_max, yaw = area
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        pose = Pose(position=Point(x=x, y=y, z=Z),
                    orientation=Quaternion(x=0, y=0, z=qz, w=qw))
        spawn_srv(cname, models[cname], '', pose, 'world')
        rospy.loginfo(f'  {cname} → ({x:.3f}, {y:.3f})')

    spawn_cones(spawn_srv)

    rospy.loginfo('完成')

if __name__ == '__main__':
    main()
