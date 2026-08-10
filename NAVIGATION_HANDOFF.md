# Gazebo 导航与抓取任务交接文档

## 1. 项目范围与当前状态

任务目标是让小车在 Gazebo 中自主导航到物块搜索区域，停车后由机械臂下压相机观察并抓取物块，随后抬臂运输到对应投放区域。

当前状态：

- 静态地图导航主链已完成并验证；
- 自动任务已经能够成功导航到修正后的第一个搜索观察点；
- `observe` 姿态下机械臂会进入激光雷达扫描平面，已实测无法正常使用 `move_base` 导航；
- 搜索现在支持按配置顺序执行多个 waypoint；每个点停车后 observe 并直接限速旋转搜索，首个发现目标立即停止后续 waypoint；该动作绕过局部规划器碰撞检查，仅用于停车后的受控观察；
- 视觉检测和机械臂几何测量已在自动任务中通过；抓取、运输和释放尚未完成端到端验证；
- 视觉基线已改为亮度连通域候选 + 类别特有文字 SIFT/RANSAC 匹配；`observe1-6` RGB 离线回归已通过；
- Gazebo 中已确认 `/cube_vision` 不退出、`/cube_vision/debug_image` 约 4--5 Hz、候选框正确、类别与置信度通过、`/cube_vision/pose` 发布且 reset 后可重新得到稳定 Pose；
- 已修复视觉坐标关键问题：Gazebo depth sensor 的 SDF 额外 `roll=pi` 未被 TF optical frame 包含，视觉节点现在将传感器点按 `sensor_to_optical_rpy=[pi,0,0]` 转入 `camera_depth_optical_frame`，并以深度帧时间戳查 TF；A 区静止实测 XY 误差约 `2.3 mm`；
- 已实现混合对位：距离目标较远时保留 `move_base` 粗对位，进入最后 `0.08 m` 后由受限 `/cmd_vel` 闭环修正 XY 和 yaw；实测平移误差从 `0.012 m` 降至约 `0.001 m`，yaw 曾收敛到 `0.054 rad`。为避免最终抓取朝向使 observe 相机离开视野，当前顺序是“保持观察朝向完成平移精调 -> 停车 fresh observe -> 抬臂后完成最终 yaw”；参数为线速度上限 `0.04 m/s`、角速度上限 `0.30 rad/s`、超时 `10 s`，待目标可见运行再次验证；
- 深度话题 `/depth_camera/depth/image_raw` 在 Gazebo 中交替出现 `rgb8` 和 `32FC1`；当前视觉节点仅过滤并使用 `32FC1`/`16UC1`/`mono16` 深度帧，未修改 URDF；
- 当前第一阶段只处理单个 `food -> cube_0` 物块；随机刷新时类别与区域不固定，测试 A 区时不能假设目标是 `cube_0`；
- daily 投放区域尚未确认，不能直接启用。

必须保持不变的规则或既有实验文件：

```text
src/car3/world/math.world
src/car3/scripts/spawn_cubes.py
src/car3/scripts/grasp_attach.py
```

其中 `math.world` 和 `spawn_cubes.py` 是规则性文件；其余文件属于既有抓取实验，不是已完成导航主链，也不作为当前新任务入口。

严禁使用旧入口：

```text
src/gazebo_nav/launch/gazebo_nav.launch
```

---

## 2. 已验证的导航基线

标准导航入口：

```text
src/gazebo_nav/launch/static_nav_sim.launch
```

默认地图：

```text
src/gazebo_map/map/ros_map_thin/ros_map_world_collision.yaml
```

主要配置：

```text
src/gazebo_nav/launch/config/static_nav/move_base.yaml
src/gazebo_nav/launch/config/static_nav/amcl.yaml
src/gazebo_nav/launch/config/static_nav/teb_local_planner_params.yaml
src/gazebo_nav/rviz/static_nav.rviz
```

导航主链保持：

- `map_server` 发布静态地图；
- AMCL 使用原始 `/scan` 定位；
- `move_base` 使用 `global_planner/GlobalPlanner`；
- `GlobalPlanner/use_dijkstra: true`；
- Global Costmap 包含 static、`/scan` obstacle 和 inflation 三层；
- Local Costmap 使用滚动窗口和 `/scan` 动态障碍；
- 默认局部规划器为 `teb_local_planner/TebLocalPlannerROS`；
- TEB 同伦类规划默认关闭。

导航数据流：

```text
地图 YAML/PGM
  -> map_server
  -> /map

/map + /scan
  -> global_costmap
  -> GlobalPlanner（Dijkstra）
  -> /move_base/GlobalPlanner/plan

/scan
  -> local_costmap obstacle_layer
  -> TEB
  -> /cmd_vel
```

当前 TEB 相关参数包括：

```yaml
max_global_plan_lookahead_dist: 1.0
min_obstacle_dist: 0.08
inflation_dist: 0.15
max_vel_x: 0.35
max_vel_y: 0.35
max_vel_trans: 0.35
max_vel_theta: 0.80
no_inner_iterations: 2
no_outer_iterations: 2
enable_homotopy_class_planning: false
```

底盘使用 Gazebo planar controller，已手动验证 `/cmd_vel.linear.y` 横向移动有效。提高 `max_vel_y` 只表示允许更高横向速度，并不会强制 TEB 优先横移；该问题当前暂缓，不继续修改已验证导航主链。

---

## 3. 环境与基础启动命令

建议按以下顺序加载环境：

```bash
source /opt/ros/noetic/setup.bash
source /home/mcx/gazebo_ws/devel/setup.bash
source /home/mcx/catkin_ws/devel/setup.bash
```

只启动静态导航基线：

```bash
roslaunch gazebo_nav static_nav_sim.launch \
  gui:=true \
  rviz:=true \
  use_teb:=true \
  teb_homotopy:=false \
  spawn_dynamic_objects:=false
```

启动包含随机物块和锥桶的导航环境：

```bash
roslaunch gazebo_nav static_nav_sim.launch \
  gui:=true \
  rviz:=true \
  use_teb:=true \
  teb_homotopy:=false \
  spawn_dynamic_objects:=true
```

`spawn_dynamic_objects:=true` 会运行既有 `spawn_cubes.py`。物块位置每次随机生成，不能把某次运行中的物块坐标写死到任务代码中。

---

## 4. 当前自动任务入口

当前新任务入口：

```text
src/car3/launch/nav_pick_place_task.launch
```

该 launch 会并发请求启动：

1. `gazebo_nav/launch/static_nav_sim.launch`；
2. 既有视觉节点 `cube_vision.py`；
3. 新任务执行器 `pick_place_executor.py`。

注意：launch 文件中的书写顺序不代表导航栈完全启动后才启动执行器。Gazebo、控制器、AMCL、move_base、视觉和执行器会并发初始化，因此执行器内部仍包含 action server 和导航数据等待。

完整启动命令：

```bash
source /home/mcx/gazebo_ws/devel/setup.bash

roslaunch car3 nav_pick_place_task.launch \
  gui:=true \
  rviz:=true \
  spawn_dynamic_objects:=true \
  target_category:=food \
  start_task:=true \
  arm_poses_verified:=true
```

关键参数：

```text
start_task=true            启动自动状态机
arm_poses_verified=true    允许执行机械臂姿态；仅表示人工确认后放行
target_category=food       第一阶段目标类别
```

夹爪参数：

```text
完全张开：1.5
抓取闭合：0.8
gripper_close_threshold：0.81
```

`0.81` 用于兼容 `grasp_attach.py` 中严格小于阈值的判断，使 `0.8` 能触发仿真附着。

---

## 5. 自动导航调试结论

### 5.1 最小 action 测试脚本

新增脚本：

```text
src/car3/scripts/test_move_base_goal.py
```

它只执行：

```text
等待 /move_base action server
-> 等待 startup_delay
-> 发送一个 map 坐标目标
-> 输出成功、失败或超时
```

不启动视觉、不控制机械臂、不控制夹爪、不清理 costmap、不自动重试。

示例：

```bash
rosrun car3 test_move_base_goal.py \
  _x:=-1.66 \
  _y:=-0.445 \
  _yaw:=0.0 \
  _startup_delay:=10.0 \
  _timeout:=60.0
```

该目标已实测导航成功，证明：

- `/move_base` action 接口正常；
- GlobalPlanner、TEB、地图与 costmap 主链正常；
- 完整任务早期失败并不是 action 接口本身造成的。

### 5.2 原始首个搜索点错误

物块生成区域 `area_a`：

```yaml
x_min: -2.10
x_max: -1.92
y_min: -0.61
y_max: -0.28
```

区域中心约为：

```text
(-2.01, -0.445)
```

最初使用：

```text
search_standoff = 0.35
search angle = 0
```

生成了错误的首个观察目标：

```text
(-2.36, -0.445, yaw=0)
```

该点经独立脚本测试失败，原因是目标过于靠近地图左侧墙体/边界。问题不是“目标发送太早”，而是把物块生成区域机械地转换成了不安全的导航观察点。

### 5.3 当前固定搜索位姿

通过 Gazebo/TF 状态快照记录了当前固定搜索位姿：

```yaml
search_pose: {x: -1.4135, y: -0.4311, yaw: 1.7201}
```

执行器不再根据 `search_areas`、`search_standoff` 和 `search_angles` 自动生成小车观察导航点。`search_areas` 仅保留为物块随机生成范围说明。

重要概念：

```text
search_areas = 物块可能随机生成的区域
```

不等于：

```text
安全的小车观察导航点
```

小车先通过 `/move_base` action 导航到固定搜索位姿，然后停车并进入新的 observe 探测姿态。

---

## 6. 机械臂姿态与激光雷达结论

当前任务姿态配置：

```yaml
navigation: [0.0, 1.6, -2.2, -1.0, 0.0]
observe:    [1.570006, 0.765642, 0.815474, 0.941225, -0.000161]
grasp:      [0.000288, 1.570820, 0.126232, 1.570326, 0.004899]
transport:  [0.0, 1.6, -2.2, -1.0, 0.0]
place:      [0.000288, 1.570820, 0.126232, 1.570326, 0.004899]
```

实测结论：

- 模型默认姿态可以导航；
- `navigation` 姿态可以导航；
- `observe` 姿态无法正常使用 `move_base` 导航；
- 原因是下压机械臂进入水平激光扫描平面，被原始 `/scan` 识别为自体障碍；
- 当前按实验要求增加 observe 下直接旋转：仅发布 `linear.x=0`、`linear.y=0`、`angular.z=0.20`，最多旋转一周，超时/漂移/检测后立即归零；
- 该直接 `/cmd_vel` 旋转绕过 move_base 碰撞检查，可能影响原始 `/scan`、AMCL 和 costmap，必须仅用于受控实验；
- `transport` 当前与 `navigation` 使用相同关节值，设计目标是在抬臂状态下运输。

因此必须遵守：

```text
navigation/默认安全姿态：允许 move_base 导航
固定搜索位姿：(-1.4135, -0.4311, yaw=1.7201)
observe：先停车、再下压；当前受控实验允许直接原地旋转一周
observe 旋转：linear.x=0、linear.y=0、angular.z=0.20，最终必须归零
抓取前若需移动：先抬回 navigation，再移动
抓取后：先抬到 transport，再导航运输
grasp/place：小车完全停止
```

第一阶段不实现 `/scan_filtered`，AMCL 和 move_base 继续使用已经验证的原始 `/scan`。

---

## 7. 当前任务执行顺序

当前期望状态机：

```text
INIT_VALIDATE
-> 等待 move_base action server
-> 等待机械臂 action server
-> 等待 joint_states
-> RESET_GRIPPER
-> 首次粗导航保持模型默认机械臂姿态
-> WAIT_NAVIGATION_READY
-> 等待 10 秒启动缓冲并确认 map/scan/TF
-> SEARCH_SOURCE
-> 通过 /move_base action 导航到 (-1.4135, -0.4311, yaw=1.7201)
-> 停车并等待稳定
-> observe 下压
-> 以 linear.x/y=0、angular.z=-0.20 顺时针直接旋转，最多一周
-> 看到任意已知方块后立即停车并确认类别
-> 非目标类别按 map 位置记录并继续剩余旋转
-> 目标类别确认后发布 detected_category 并进入后续流程
-> 检测到目标或旋转完成后发布零速度
-> 检测到目标后进行机械臂几何测量与视觉对位
-> grasp
-> transport
-> 导航到投放区域
-> place/release
-> SUCCESS
```

当前已确认到：

```text
SEARCH_SOURCE -> (-1.66, -0.445, pi) 粗导航成功
```

当前固定搜索位姿、顺序多 waypoint 搜索、observe 下直接旋转和混合粗/精对位逻辑已写入代码并完成部分 Gazebo 实测。最近一次多 waypoint 运行中，首点未命中后成功导航到第二点并发现 `food`；后续需复测精调后的抓取附着。

---

## 8. 常用运行检查

检查节点：

```bash
rosnode list
```

导航基线应包含：

```text
/gazebo
/map_server
/amcl
/move_base
/robot_state_publisher
```

完整任务还应包含：

```text
/cube_vision
/pick_place_executor
```

检查规划器：

```bash
rosparam get /move_base/base_global_planner
rosparam get /move_base/base_local_planner
```

预期：

```text
global_planner/GlobalPlanner
teb_local_planner/TebLocalPlannerROS
```

检查全局 costmap 插件：

```bash
rosparam get /move_base/global_costmap/plugins
```

预期包括：

```text
static_layer
obstacle_layer
inflation_layer
```

检查任务状态：

```bash
rostopic echo /pick_place_executor/state
```

检查导航 action：

```bash
rostopic echo /move_base/status
rostopic echo /move_base/goal
rostopic echo /move_base/result
```

检查定位和激光：

```bash
rostopic echo /amcl_pose
rostopic hz /scan
rosrun tf tf_echo map base_footprint
```

RViz 建议显示：

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

---

## 9. 已知限制与后续工作

下一阶段应按以下顺序继续，不要同时修改导航和抓取：

1. 在 B/C 区和多个静止视角重复校准脚本，确认视觉 XY 误差和高度误差；
2. 在目标可见的运行中复测 `FINE_ALIGN_TO_GRASP`：进入约 8 cm 后只能发布受限 `/cmd_vel`（当前线速度上限 `0.04 m/s`、角速度上限 `0.30 rad/s`），确认 XY/yaw 都收敛到阈值；
3. 验证车体对位前先抬回 `navigation`；平移精调停车后会尝试 observe、reset、获取 fresh Pose，若相机丢帧则沿用最后稳定 map 坐标，再抬臂完成最终 yaw；
4. 验证 `grasp_attach/ready`、夹爪 `0.8` 和正确的 `attached_model`；
5. 验证抓取后切换 `transport` 时物块不丢失；
6. 验证原始 `/scan` 下运输导航；
7. 验证 food 投放区域释放和 Gazebo 模型位置稳定性；
8. 最后再确认 daily 投放区域并扩展到三类物块。

当前不要做：

- 不使用旧 `gazebo_nav.launch`；
- 不修改规则文件或既有抓取实验文件；
- 不在 `observe/grasp/place` 姿态下调用正常 move_base 导航；
- 不把某次随机物块坐标写死；
- 不声称已支持任意末端目标 IK；
- 不在视觉与抓取尚未验证前扩展三物块完整流程；
- 不把 TEB 横向移动偏好调试与抓取调试混在同一轮修改中。

---

## 10. 当前交接验收状态

已完成：

- 标准导航入口、地图、Dijkstra、Global Costmap `/scan` 和 TEB 基线；
- RViz 手动导航；
- `/move_base` action 最小脚本导航；
- 自动任务修正首个搜索目标后的粗导航；
- 默认姿态与 `navigation` 姿态导航验证；
- `observe` 姿态不可使用 `move_base` 导航的实测确认；
- 固定搜索位姿和快照 observe 五关节值已记录；
- arm-down 直接限速旋转扫描逻辑已实现并完成自动任务实测；
- arm-down 停车、抬臂后导航的任务约束。
- Gazebo A 区视觉坐标校准：`cube_0` 的 map 真值约 `(-1.9643,-0.3937,0.0104)`，视觉稳定输出约 `(-1.9650,-0.3959,0.0077)`，XY 误差 `2.4 mm`、Z 误差 `-2.7 mm`；
- 视觉校准脚本已修复并可运行：`src/car3/scripts/calibrate_vision_ground_truth.py`。

尚未完成：

- 观察点相机视野覆盖验证；
- B/C 区和多视角视觉坐标回归；当前仅完成 A 区单个静止视角；
- 深度图像话题混合编码的长期稳定处理；当前仅在视觉订阅端过滤有效深度编码，未修改 URDF；
- 视觉闭环对位：粗对位完成，低速 `/cmd_vel` 精调已实测 XY/yaw 进入阈值附近；最终 fresh observe 顺序已调整，丢帧时允许沿用最后稳定 map 坐标，抓取附着尚待验证；
- 随机搜索覆盖：已加入 `area_a -> area_b -> area_c` 的顺序搜索配置；实测首点未命中时会导航到第二点并成功发现目标，首点命中时直接结束搜索；第三点仍需独立验证视野和导航安全性；
- 单物块抓取附着；
- 抬臂运输；
- 投放与释放验证；
- daily 区域确认；
- 三类物块完整闭环。
