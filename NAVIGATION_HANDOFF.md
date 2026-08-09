# Gazebo Navigation Handoff

## Scope

This document covers the completed navigation stack. Pick/place manipulation is still experimental and is not part of the navigation baseline.

Do not modify these files while continuing navigation work:

- `src/car3/world/math.world`
- `src/car3/scripts/spawn_cubes.py`

The current `spawn_cubes.py` checksum is:

```text
27c21cf4f31f3dad34176962abdb16d5be02122455e60e9d8320880a29780c92
```

## Canonical Navigation Entry

Use this launch file for the completed navigation task:

```text
src/gazebo_nav/launch/static_nav_sim.launch
```

It starts:

- Gazebo and the car through `gazebo_map/launch/slam_sim.launch`
- `map_server`
- AMCL by default
- `move_base`
- RViz when enabled
- optional random cubes and cones

The current default map is:

```text
src/gazebo_map/map/ros_map_thin/ros_map_world_collision.yaml
```

The navigation parameters are loaded from:

```text
src/gazebo_nav/launch/config/static_nav/move_base.yaml
```

## Required Environment

Source the workspaces in this order:

```bash
source /opt/ros/noetic/setup.bash
source /home/mcx/gazebo_ws/devel/setup.bash
source /home/mcx/catkin_ws/devel/setup.bash
```

The last source is important because `catkin_ws` provides the navigation and controller packages used by the launch files.

## Recommended Commands

### Static navigation baseline

Use this to test the static map and walls without random objects:

```bash
roslaunch gazebo_nav static_nav_sim.launch gui:=true rviz:=true use_teb:=true teb_homotopy:=false spawn_dynamic_objects:=false
```

### Current navigation with random dynamic objects

Use this to test the current dynamic-obstacle configuration:

```bash
roslaunch gazebo_nav static_nav_sim.launch gui:=true rviz:=true use_teb:=true teb_homotopy:=false spawn_dynamic_objects:=true
```

`spawn_dynamic_objects:=true` starts the existing `spawn_cubes.py` node. It generates random cubes and cones. Do not use cube positions from one run as fixed positions for another run.

## Navigation Data Flow

```text
math.world
  -> Gazebo walls and robot

map YAML and PGM
  -> map_server
  -> /map

/map + /scan
  -> global_costmap
  -> GlobalPlanner
  -> /move_base/GlobalPlanner/plan

/scan
  -> local_costmap obstacle_layer
  -> TEB local planner
  -> /cmd_vel
```

The current global costmap includes:

```text
static_layer
obstacle_layer from /scan
inflation_layer
```

This means cones and other laser-visible objects are included in global replanning. The planner frequency is currently `1.0 Hz`.

The local costmap remains a rolling `2.5 m x 2.5 m` map with `/scan` obstacle marking and inflation.

## Current Planner Configuration

Global planner:

```yaml
base_global_planner: global_planner/GlobalPlanner
GlobalPlanner/cost_factor: 0.5
GlobalPlanner/use_dijkstra: true
GlobalPlanner/use_quadratic: true
GlobalPlanner/use_grid_path: false
```

TEB is enabled at launch with:

```text
use_teb:=true
```

Current relevant TEB settings:

```yaml
max_global_plan_lookahead_dist: 1.0
min_obstacle_dist: 0.08
inflation_dist: 0.15
max_vel_x: 0.35
max_vel_y: 0.05
max_vel_theta: 0.80
no_inner_iterations: 2
no_outer_iterations: 2
enable_homotopy_class_planning: false
```

To test homotopy as an isolated comparison, change only the launch argument:

```bash
roslaunch gazebo_nav static_nav_sim.launch use_teb:=true teb_homotopy:=true spawn_dynamic_objects:=true
```

Do not mix this navigation A/B test with manipulation changes.

## RViz Topics

Useful RViz displays:

```text
/map
Global Costmap
Local Costmap
LaserScan: /scan
Global Plan: /move_base/GlobalPlanner/plan
TEB Global Plan: /move_base/TebLocalPlannerROS/global_plan
TEB Local Plan: /move_base/TebLocalPlannerROS/local_plan
RobotModel
```

Send a goal using RViz `2D Nav Goal`. The goal frame should be `map`.

## Runtime Checks

Check the active nodes:

```bash
rosnode list
```

Expected navigation nodes include:

```text
/gazebo
/map_server
/amcl
/move_base
/robot_state_publisher
```

Check the active planner:

```bash
rosparam get /move_base/base_global_planner
rosparam get /move_base/base_local_planner
```

With TEB enabled, expected output includes:

```text
global_planner/GlobalPlanner
teb_local_planner/TebLocalPlannerROS
```

Check the global costmap plugins:

```bash
rosparam get /move_base/global_costmap/plugins
```

Expected plugins are:

```text
static_layer
obstacle_layer
inflation_layer
```

Check navigation action state:

```bash
rostopic echo /move_base/status
```

Check robot localization:

```bash
rostopic echo /amcl_pose
```

## Important Launch Distinction

The following file is **not** the canonical static navigation entry:

```text
src/gazebo_nav/launch/gazebo_nav.launch
```

It is an older/alternative launch file. It:

- hard-codes `math.yaml`
- uses a different parameter directory
- uses DWA directly
- expects a different RViz path
- should not be used to reproduce the current static navigation result

Use `static_nav_sim.launch` instead.

## Files Used by Completed Navigation

Primary launch files:

```text
src/gazebo_nav/launch/static_nav_sim.launch
src/gazebo_map/launch/slam_sim.launch
```

Navigation configuration:

```text
src/gazebo_nav/launch/config/static_nav/move_base.yaml
src/gazebo_nav/launch/config/static_nav/amcl.yaml
src/gazebo_nav/launch/config/static_nav/teb_local_planner_params.yaml
src/gazebo_nav/rviz/static_nav.rviz
```

Map files:

```text
src/gazebo_map/map/ros_map_thin/ros_map_world_collision.yaml
src/gazebo_map/map/ros_map_thin/ros_map_world_collision.pgm
```

The actual PGM name should always be checked from the selected map YAML before changing maps.

## Files That Are Not Needed for Navigation Handoff

These files belong to object generation, perception, or manipulation experiments:

```text
src/car3/scripts/spawn_cubes.py
src/car3/scripts/cube_vision.py
src/car3/scripts/grasp_attach.py
src/car3/scripts/test_observe_grasp.py
src/car3/scripts/gripper_mimic.py
src/car3/config/pick_place_task.yaml
src/car3/launch/pick_place_task.launch
```

Their roles are:

- `spawn_cubes.py`: random cube and cone generation; immutable for this project
- `cube_vision.py`: RGB/template recognition and experimental pose output
- `grasp_attach.py`: Gazebo grasp simulation and attachment status
- `test_observe_grasp.py`: experimental observe-to-grasp test only
- `gripper_mimic.py`: publishes complete mimic joint states
- `pick_place_task.launch`: navigation plus perception/manipulation experiment entry; not the navigation baseline

Do not include `test_observe_grasp.py` in a navigation-only handoff or use it as a navigation health check.

## Navigation Handoff Acceptance

A navigation-only handoff is complete when:

1. `static_nav_sim.launch` starts with the required workspace overlay order.
2. AMCL publishes a valid pose.
3. `move_base` uses GlobalPlanner and the intended TEB/DWA local planner.
4. RViz displays global and local costmaps.
5. A normal RViz goal reaches its target.
6. With dynamic objects enabled, laser-visible cones appear in the configured costmaps and the global plan can be replanned around them.
7. No changes are made to `math.world` or `spawn_cubes.py`.

Manipulation work should start only after this baseline has been reproduced independently.
