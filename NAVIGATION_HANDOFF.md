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
- TCP/ROS 双向桥接已完成通信测试：本机作为客户端连接 `192.168.10.246:9000`，桥接 `/cube_category` 话题和 `/gazebo_success` 参数；TCP 已不再是当前阻塞项。
- 下一阶段主线改为任务耗时优化，优先处理“搜索物块”和“确认夹取”两个阶段；当前尚未记录分阶段耗时基线，不能先凭感觉缩短超时。

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
3. 新任务执行器 `pick_place_executor.py`；
4. 默认启用的 TCP/ROS 桥接节点 `tcp_ros_bridge.py`。

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
tcp_bridge=true            启动 TCP/ROS 桥接，当前默认开启
tcp_remote_host=192.168.10.246
tcp_remote_port=9000
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
/tcp_ros_bridge
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

下一轮当前主线是速度优化，应按以下顺序进行：

1. 给状态切换增加耗时记录，至少分别统计 `SEARCH_SOURCE/ROTATE_SEARCH/CONFIRM_CUBE` 和 `PREPARE_GRASP/GRASP`；先跑出基线，再改参数或逻辑；
2. 优先优化搜索物块阶段：区分 waypoint 导航、`search_settle_time`、机械臂切换、旋转扫描、`area_search_timeout` 和类别确认各自耗时；继续保持“首个点发现目标后立即跳过剩余 waypoint”；
3. 再优化确认夹取阶段：区分 fresh Pose、机械臂进入 `grasp`、`grasp_attach/ready`、夹爪闭合、`attached_model` 确认和闭合保持各自耗时；
4. 每次只修改一类等待或动作参数，并用相同启动条件重复测试，确认提速没有引入漏检、误抓、附着失败或物块丢失；
5. 超时参数只是失败上限；对于已经满足条件就提前返回的等待，单纯缩短超时不会加快成功路径，必须以分阶段日志定位真实等待点。

仍需保留的功能验证 backlog：

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
- TCP client 已能连接 `192.168.10.246:9000`，TCP/ROS 桥接通信测试已由用户确认通过。

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

---

## 11. TCP/ROS 桥接交接

### 11.1 当前实现

脚本：

```text
src/car3/scripts/tcp_ros_bridge.py
```

安装配置已加入：

```text
src/car3/CMakeLists.txt
```

启动配置已加入：

```text
src/car3/launch/nav_pick_place_task.launch
```

通信方向和角色：

```text
本机 gazebo_ws：TCP client
远端 192.168.10.246：必须运行 TCP server
默认端口：9000
协议：UTF-8 JSON Lines，每条 JSON 后必须有换行符 \n
```

双向桥接对象：

```text
/cube_category   std_msgs/String 话题
/gazebo_success  ROS 参数
```

远端可发送以下消息来更新本机 ROS：

```json
{"type":"cube_category","value":"food"}
{"type":"gazebo_success","value":1}
```

本机向远端发送的消息还会分别包含 `topic` 或 `param` 字段；远端应按 `type` 和 `value` 处理，并忽略不需要的额外字段。

### 11.2 当前同步语义

- `/gazebo_success` 默认每 `0.2 s` 检查一次，启动时发送当前值；参数不存在时按 `0` 处理；
- 后续只有参数值变化时才发送，不是周期心跳；
- 收到远端 `gazebo_success` 消息后会设置本机 `/gazebo_success`；
- TCP 断线后默认每 `2 s` 自动重连；断线期间检测到的本地变化暂存在最多 100 条的发送队列中；
- 每次重连成功后不会无条件重发当前 `/gazebo_success`，也没有 ACK/失败重传，因此目前不能声称是强可靠状态同步；
- `/cube_category` 收发均已实现，并带有简单的本机回环抑制。

如果下一阶段要求远端重启后必然得到最新状态，最小修改应是：每次 TCP 连接成功时主动发送一次当前 `/gazebo_success`。是否还需要周期心跳或应用层 ACK，应由对端协议要求决定，不要先行增加。

### 11.3 联调结果

当前状态：

```text
本机：192.168.10.217/24，接口 wlp2s0
目标：192.168.10.246
TCP 192.168.10.246:9000：通信测试通过
```

此前曾出现 `Connection refused`，当时目标主机可 ping，但 `9000` 没有可用监听服务。该问题现已排除，不再是当前阻塞项。远端具体修复动作和本轮测试报文未记录，因此不要在交接文档中推断其防火墙或服务配置。

如后续再次无法连接，先检查远端监听：

```bash
ss -ltnp | grep ':9000'
```

TCP server 应监听 `0.0.0.0:9000` 或 `192.168.10.246:9000`，不能只监听 `127.0.0.1:9000`。如果实际端口不是 9000，启动时传入：

```bash
roslaunch car3 nav_pick_place_task.launch \
  tcp_bridge:=true \
  tcp_remote_host:=192.168.10.246 \
  tcp_remote_port:=实际端口
```

本机可先验证端口：

```bash
nc -vz -w 3 192.168.10.246 9000
```

正常情况下任务日志应出现：

```text
TCP bridge connected
```

### 11.4 联调验收步骤

本机到远端：

```bash
rostopic pub --once /cube_category std_msgs/String "data: 'food'"
rosparam set /gazebo_success 1
```

远端应收到两条以换行分隔的 JSON 消息。

远端到本机：

```json
{"type":"cube_category","value":"electronics"}
{"type":"gazebo_success","value":0}
```

本机检查：

```bash
rostopic echo /cube_category
rosparam get /gazebo_success
```

当前验证状态：

- `tcp_ros_bridge.py` Python 语法检查通过；
- launch XML 检查通过；
- `car3` 单元测试 `3/3` 通过；
- `catkin_make --pkg car3 -j2` 通过；
- TCP 通信测试已由用户确认通过。

TCP 当前进入回归保护状态，不作为下一轮优化重点。修改搜索或夹取逻辑后，仅需确认桥接节点仍能连接，并且 `/cube_category` 与 `/gazebo_success` 的既有通信没有回归。

---

## 12. 下一轮任务：搜索与夹取确认速度优化

### 12.1 优化范围

只优化以下两段：

```text
搜索物块：SEARCH_SOURCE -> ROTATE_SEARCH/OBSERVE_* -> CONFIRM_CUBE
确认夹取：PREPARE_GRASP -> GRASP -> attached_model 确认
```

暂时不要同时调整 TEB、地图、视觉坐标模型、运输路线或投放逻辑。当前没有给出目标总耗时或期望提升比例，下一轮必须先测量基线，不能宣称已经提速。

### 12.2 搜索阶段需要测量的节点

```text
到达每个 waypoint 的时间
停车后的 search_settle_time（当前 0.5 s）
navigation -> observe 的机械臂动作与稳定时间
首帧有效检测和类别确认时间
旋转搜索时间、旋转角度和是否发生非目标确认停顿
空 waypoint 的 area_search_timeout（当前 6.0 s）
首点命中时是否确实跳过后续 waypoint
```

搜索优化必须继续满足：目标一旦确认立即停车、发布零速度并停止后续 waypoint；`observe` 下的直接旋转仍受漂移、角度和超时限制，不能为了提速绕开现有安全归零逻辑。

### 12.3 夹取确认阶段需要测量的节点

```text
fresh Pose 样本收集：默认 3 个样本、最长 3.0 s
进入 grasp 姿态及关节稳定时间
等待 grasp_attach/ready 和 attach_offset 进入阈值的时间
夹爪 close 到到位的时间
等待 GRASPING 且 attached_model 正确的时间（attach_timeout 当前 5.0 s）
再次发送 hold 到到位的时间
```

`PREPARE_GRASP` 和 `_wait_attach()` 都会在条件满足时提前返回；只有实测发现有效事件到达慢、重复动作或固定等待占时，才修改对应逻辑。不得通过跳过 `ready`、`attach_offset`、`GRASPING` 或 `attached_model` 检查来换取速度。

### 12.4 验收原则

每次修改后至少记录：

```text
搜索阶段耗时
确认夹取阶段耗时
找到的 category
attached_model
最终 grasp_state
是否进入后续 transport
```

验收必须同时满足：耗时相对基线下降；首点命中短路仍有效；目标类别正确；夹取确认成功；抬到 `transport` 后物块不丢失；失败路径仍会停车并给出明确日志。

---

## 13. 2026-08-11 搜索速度优化实测

### 13.1 本轮修改

`pick_place_executor.py` 增加 wall-clock 分阶段日志，记录：

```text
搜索：waypoint 导航、停车稳定、observe 动作、旋转/检测、非目标确认、总耗时
夹取：grasp 动作、ready+attach_offset、夹爪 close、GRASPING+attached_model、hold、总耗时
```

最终只保留一个速度参数修改：

```yaml
search_rotation_speed: -0.20 -> -0.25
```

没有修改任何 timeout，也没有修改类别、ready、attach_offset、`GRASPING` 或 `attached_model` 检查。旋转的一周角度上限、漂移限制、超时和 `finally` 零速度逻辑保持不变。

### 13.2 基线

启动条件：

```bash
roslaunch car3 nav_pick_place_task.launch \
  gui:=false rviz:=false spawn_dynamic_objects:=true \
  arm_poses_verified:=true tcp_bridge:=false
rostopic pub --once /cube_category std_msgs/String "data: 'food'"
```

物块仍由 `spawn_cubes.py` 随机刷新。基线中 `cube_0/food` 位于 A 区约 `(-1.975, -0.339)`；粗搜索首点命中，未访问后续固定 waypoint：

```text
search.coarse.navigation       29.552 s
search.coarse.settle            0.500 s
search.coarse.observe_arm       3.495 s
search.coarse.rotation          2.369 s / 0.42 rad
search.total                   35.928 s

grasp.prepare.arm               3.371 s
grasp.prepare.ready_offset      0.000 s
grasp.prepare.total             3.373 s
grasp.close                     0.728 s
grasp.attach_confirm            0.220 s  (GRASPING, cube_0)
grasp.hold                      0.000 s
grasp.total                     0.955 s
```

该轮进入 `PARKED -> SUCCESS`，抬到 transport 并完成运输时附着未丢失。

基线表明 `ready/attach_offset` 和 attach 确认都在条件满足后立即提前返回；缩短它们的 timeout 不会加快成功路径。粗搜索旋转则随扫描角度直接增加耗时，实测约 `5.64 s/rad`。

### 13.3 优化结果

最终复测使用相同 launch 参数并再次随机刷新；`cube_0/food` 位于 A 区约 `(-1.966, -0.371)`。结果：

```text
search.coarse.navigation       31.107 s
search.coarse.settle            0.500 s
search.coarse.observe_arm       3.484 s
search.coarse.rotation          0.453 s / 0.10 rad
search.total                   35.554 s

grasp.prepare.arm               3.315 s
grasp.prepare.ready_offset      0.000 s
grasp.prepare.total             3.319 s
grasp.close                     0.674 s
grasp.attach_confirm            0.259 s  (GRASPING, cube_0)
grasp.hold                      0.000 s
grasp.total                     0.938 s
```

旋转归一化耗时约 `4.53 s/rad`，相对基线约下降 `19.7%`；由于日志角度仅保留两位小数，该百分比只作为本轮量级估计。尽管最终轮导航比基线多 `1.555 s`，搜索总耗时仍从 `35.928 s` 降到 `35.554 s`。目标确认后立即结束搜索，日志中没有任何 `OBSERVE_AREA_*`，证明没有继续后续 waypoint。最终类别为 `food`，attach 确认为 `GRASPING/cube_0`，并进入 `PARKED -> SUCCESS`，transport 后物块未丢失。

另一次保持 `-0.20 rad/s` 的随机 C 区运行中，旋转先确认 `daily` 非目标，再在 `3.25 rad` 处确认 `food`，说明非目标类别检查和继续搜索逻辑仍工作。该轮曾试验把 grasp 轨迹从 `3.0 s` 改为 `2.0 s`；虽然 `PREPARE_GRASP` 降到 `2.926 s` 且 attach 成功，但进入 transport 后附着丢失，因此该试验不满足验收，已完全恢复为 `3.0 s`，不计作优化收益。

### 13.4 验证与遗留问题

已验证：

- 标准入口为 `nav_pick_place_task.launch`，未使用旧 `gazebo_nav.launch`；
- `math.world`、`spawn_cubes.py`、`grasp_attach.py` 未修改，随机坐标未写入任务代码；
- 首点目标确认后跳过后续 waypoint；搜索旋转检测后由原有 `finally` 发布零速度；
- 类别、fresh Pose 流程、ready、attach_offset、`GRASPING`、正确 `attached_model` 均保留；
- 夹取后进入 transport、停车区域和 `SUCCESS`；
- TCP 测速时关闭以隔离外部网络波动，桥接代码和接口未修改。

遗留问题：当前 `-0.25 rad/s` 只完成一次完整成功复测；下一步应在随机 A/B/C 分布下重复运行，重点确认更长旋转角度时不漏检，并按角度统计多轮均值。C 区那次 transport 脱离发生在已撤销的 grasp 轨迹试验中，后端日志未给出独立原因；在再次尝试缩短 grasp 动作前，应先复现并定位脱离条件，不能继续直接压缩动作时长。

---

## 14. 2026-08-12 搜索区固定观察点直接移动

本节记录第一版“底盘平移同时调整观察 yaw”的对照数据；最终实现已进一步改为“底盘保持 yaw 平移、机械臂 joint1 补偿观察方向”，以第 15 节为准。

### 14.1 修改范围

初始出生点到 `coarse_search_pose` 仍使用 `move_base`，因为“搜索区内部无障碍”不能推导出整条初始路线无障碍。只有已经到达粗搜索区后，A/B/C 固定观察点之间改用 map TF 闭环 `/cmd_vel`：

```text
最大直接移动距离：0.30 m
线速度上限：0.15 m/s
角速度上限：0.50 rad/s
超时：10.0 s
控制频率：沿用 fine_align_rate=20 Hz
位置/yaw 验收：沿用 0.012 m / 0.04 rad
```

超过 `0.30 m` 时继续使用 `move_base`。直接移动前由既有状态机保证机械臂已经回到 `navigation`；移动采用实时 `map -> base_footprint` 误差闭环，而不是按固定时间盲走。正常到达、超时、TF/运行异常和 ROS shutdown 都经过 `finally` 发布零速度。

### 14.2 move_base 基线与直接移动结果

使用标准 `nav_pick_place_task.launch`、随机物块和同一组配置 waypoint。move_base 基线通过 `/move_base_simple/goal` 逐点发送并确认每次结果均为 `Goal reached`：

```text
转换             move_base       直接移动       节省
coarse -> area_a   3.976 s         1.767 s       55.6%
area_a -> area_b   9.077 s         5.744 s       36.7%
area_b -> area_c   7.671 s         5.547 s       27.7%
合计              20.724 s        13.058 s       37.0% / 7.666 s
```

直接移动终点日志：

```text
area_a: xy_error=0.012 m, yaw_error= 0.010 rad
area_b: xy_error=0.000 m, yaw_error=-0.039 rad
area_c: xy_error=0.002 m, yaw_error=-0.039 rad
```

为确定性覆盖三个 waypoint，验证运行临时将粗旋转角度设为 `0.01 rad`，并选择本轮随机位于 C 区的 `electronics/cube_2` 为目标。运行依次在 A 区确认 `food`、B 区确认 `daily`、C 区确认目标 `electronics`，随后 `search.total` 以 `source=area_c` 正常返回。测试后已将正式 `search_rotation_angle` 恢复为 `6.283185`，物块随机坐标未写入代码。

### 14.3 保持不变与后续

- 粗搜索初始导航、旋转搜索、目标首见短路、类别确认、视觉 reset 和每点 observe 流程不变；
- 每次直接移动结束均停车，之后才执行 `search_settle_time` 和 observe；
- 地图、生成脚本、attach 后端、导航配置、视觉、夹取和 TCP 均未修改；
- 本轮验证的是搜索 waypoint 移动与分类链，没有重复执行其后的 C 区抓取运输；抓取运输状态仍以第 13 节成功运行作为当前基线。

后续如果提高 `0.15 m/s` 或扩大 `0.30 m` 范围，必须重新检查路径净空和停车误差；不能把这种直接移动扩展到出生点到搜索区、运输路线或投放区域。

---

## 15. 2026-08-12 麦克纳姆纯平移与机械臂转向

### 15.1 最终策略

搜索区内不再让底盘转到每个 waypoint 的观察 yaw：

```text
底盘：保持进入搜索区时的 yaw，使用 linear.x + linear.y 闭环平移
机械臂：到点后在原 3 s observe 动作中同时调整 arm_joint1
观察方向：arm_joint1_offset = waypoint_yaw - actual_base_yaw（选择关节限位内的等价角）
```

底盘仍用小角速度闭环抵消 yaw 漂移，但目标是保持原朝向，不是转向 waypoint yaw。observe 的其余四个关节、动作时长、稳定检查、视觉 reset 和类别确认不变。距离超过 `0.30 m` 回退 move_base 时，底盘仍到达原 waypoint yaw，机械臂不加补偿。

第一次验证在 C 区发现等价角选择问题：`+3.036 rad` 叠加 nominal joint1 后超出限位，运行被既有限位保护明确终止。随后改为从 `delta + {-2pi, 0, 2pi}` 中选择最终 joint1 位于 `[-3.14, 3.14]` 的最小绝对补偿；C 区选到 `-3.263 rad`，最终 joint1 约 `-1.69 rad`，动作与视觉确认成功。失败版本未保留。

### 15.2 最终耗时

同样通过临时 `search_rotation_angle=0.01` 确定性覆盖 A/B/C；测试后已恢复正式值 `6.283185`：

```text
转换             move_base    底盘边移边转    纯平移+机械臂转向
coarse -> area_a   3.976 s       1.767 s          1.816 s
area_a -> area_b   9.077 s       5.744 s          1.716 s
area_b -> area_c   7.671 s       5.547 s          3.580 s
合计              20.724 s      13.058 s          7.112 s
```

最终策略相对 move_base 节省 `13.612 s / 65.7%`，相对底盘边移边转再节省 `5.946 s / 45.5%`。三段结束时底盘 yaw 保持误差日志均约为 `0.000 rad`，XY 误差分别为 `0.012/0.012/0.012 m`。

机械臂补偿与视觉结果：

```text
area_a joint1_offset= 0.027 rad -> 确认 electronics 非目标
area_b joint1_offset=-1.693 rad -> 本轮 6 s 内无检测
area_c joint1_offset=-3.263 rad -> 确认 food 目标，source=area_c
```

此前一次相同方向补偿运行中，A/B 分别正确确认 `food/electronics`；因此三个区域都已分别出现正确识别，但还没有一轮在随机边缘位置同时命中 A/B/C。本轮 B 区物块位于区域边缘附近且未检出，需作为视觉覆盖回归项继续重复随机测试，不能把单轮搜索总耗时当作最终均值。

---

## 16. 2026-08-12 固定搜索确认与夹爪阈值复测

固定区域搜索现在对每个候选执行两个额外视觉 reset；初始窗口加两个确认窗口中，类别至少需要两票。若目标类别与任一其他类别在这些独立窗口中冲突，则不允许用 `2:1` 开始抓取，而是继续下一视角。不同机械臂视角在 5 cm 内看到同一 Pose 却给出不同类别时也判为冲突。实测曾出现 B 区同一 daily 物块跨视角翻为 electronics；该保护阻止了错误抓取。

干净随机复测中物块分布为 A 区 daily、B 区 electronics、C 区 food。临时把粗旋转限制为 `0.01 rad` 后，任务在 A 区拒绝了无法独立复现的单次候选，在 B 区以 `3/3` 确认 electronics 非目标，最后在 C 区中心静止视角约 `1.2 s` 找到 food 并以 `3/3` 确认。正式 YAML 的完整粗旋转参数未修改。

夹爪实测表明，发送 `0.8` 后物块接触会使 `r_joint` 停在约 `0.8402`。因此 attach 后端的 launch 参数使用 `close_threshold=0.86`；该次手动验证得到：

```text
/grasp_attach/ready: true
/grasp_attach/state: GRASPING
/grasp_attach/attached_model: cube_0
r_joint: 0.8402
```

原 `open_threshold=0.89` 会在机械臂抬升时因关节瞬态波动误释放，表现为物块夹到半空掉落。任务要求抓取后不放下，因此 launch 将释放阈值设为 `1.2`，只让显式 `1.5` 打开命令释放。执行器关闭/保持夹爪时也允许已经进入 `GRASPING` 作为关节等待完成条件，随后仍严格校验目标类别对应的 `attached_model`；transport 抬升后和停车完成后继续调用 `_check_attachment()`。

---

## 17. UDP/ROS 桥接

脚本 `src/car3/scripts/udp_ros_bridge.py` 提供与 TCP 桥相同的双向同步对象：

```text
/cube_category   std_msgs/String 话题
/gazebo_success  ROS 参数
```

UDP 桥默认关闭，避免与默认开启的 TCP 桥重复转发。启用 UDP 时建议显式关闭 TCP：

```bash
roslaunch car3 nav_pick_place_task.launch \
  tcp_bridge:=false \
  udp_bridge:=true \
  udp_bind_host:=0.0.0.0 \
  udp_bind_port:=9000 \
  udp_remote_host:=192.168.10.246 \
  udp_remote_port:=9000
```

每个 UDP 数据报包含一个 UTF-8 JSON 对象；同一数据报内也兼容多行 JSON。协议字段与 TCP 桥一致：

```json
{"type":"cube_category","topic":"/cube_category","value":"food"}
{"type":"gazebo_success","param":"/gazebo_success","value":1}
```

收到 `cube_category` 后发布本机话题，收到 `gazebo_success` 后设置本机参数；本机话题消息和参数变化会发送到配置的远端地址。`/gazebo_success` 默认每 `0.2 s` 轮询且只在值变化时发送。UDP 无连接、无 ACK、无可靠重传；需要可靠交付时继续使用 TCP 桥或由外部协议实现确认与重发。

---

## 18. 抓取后运输起点的代价地图刷新

一次 daily/C 区运行中抓取成功且保持 `GRASPING/cube_1`，但两个停车朝向都在运输起点失败。move_base 依次报告局部轨迹不可行、全局规划失败，以及当前位置存在潜在碰撞。该位置在静态地图中仍有约 `0.35 m` 净空，失败特征符合地面物块被抬走后，global/local obstacle layer 仍短暂保留原激光障碍点，使机器人被判定处于碰撞区。

执行器现在在 `PREPARE_TRANSPORT` 中完成抬臂并确认附着，然后调用既有 `/move_base/clear_costmaps`，等待 global/local costmap 各发布至少一次新更新，再进入 `NAVIGATE_TO_PARKING`。该刷新只针对动态 obstacle layer；静态地图由 static layer 恢复，不修改地图、规划器或抓取后端。日志应出现：

```text
pick_place state: PREPARE_TRANSPORT
costmaps refreshed after pickup: global_updates=... local_updates=...
pick_place state: NAVIGATE_TO_PARKING
```

任务启动时原有的固定 `navigation_startup_delay: 10.0` 已移除。执行器进入 `WAIT_NAVIGATION_READY` 后立即检查 map、scan、map 到底盘 TF 和位姿稳定性；真实导航数据未就绪时仍会等待，避免用固定时间盲等，也不会在导航尚未可用时发送目标。

---

## 19. 抓取准备失败自动重试

`PREPARE_GRASP` 中若 `/grasp_attach/ready` 或 `attach_offset` 未进入保守夹取框，执行器不再立即退出。因为此时夹爪尚未闭合且没有附着，状态机会进入 `RETRY_GRASP_ALIGN`，重新观察并再次执行精对位，默认最多重试 2 次。

近距离相机仍丢帧时，执行器使用抓取后端在失败姿态下已经发布的实时 `attach_offset` 重建一次目标 map 坐标。该回退目标必须在上一次视觉目标 `0.12 m` 内，否则拒绝使用，避免误跟随其他物块。只有准备阶段错误会自动重试；夹爪已经闭合后的 attach 型号错误或运输中丢失仍按原有失败恢复处理。

预期日志：

```text
grasp preparation failed: ... ready=False offset=(...)
using grasp offset fallback for retry: ...
pick_place state: RETRY_GRASP_ALIGN
retrying grasp alignment 1/2
```

---

## 20. 粗对位改为一次直达

原粗对位每次最多移动 `0.15 m`，一次典型运行从 `0.435 m` 误差经过 `0.285/0.123/0.000 m`，产生三次 move_base 和多次重复观察。现在 `max_align_step` 调整为 `1.0 m`，正常情况下直接导航到第一次视觉估计的抓取前位姿，到达后只重新观察一次，再进入最后 `8 cm` 内的受限 `/cmd_vel` 精调。

`max_align_iterations` 保留为 2：如果到达后的 fresh Pose 显示误差仍大于 `0.08 m`，允许一次额外 move_base 粗修正；如果相机丢帧，则沿用最后稳定目标并由既有 `PREPARE_GRASP` 自动重试保护兜底。ready、attach_offset、附着型号和运输检查均未放宽。

---

## 21. 任务失败后保持类别并在失败点重搜

局部 `RETRY_GRASP_ALIGN` 两次仍不能满足保守夹取框，或者闭爪、运输和停车阶段发生异常时，执行器不再返回起点或清空 `target_category`。它保持 `/gazebo_success=0`，发布 `RETRYING:<原因>`，打开夹爪并将机械臂恢复到 navigation 姿态，然后进入 `SEARCH_NEAR_FAILURE`。只有安全恢复动作本身连续三次失败时，才明确发布 `FAILED` 并停车退出，避免无限停留在 `RECOVER_FAILURE`。

`SEARCH_NEAR_FAILURE` 记录停车后的实时 `map -> base_footprint` 位姿，依次检查该点及其前后左右 `0.12 m` 的有界位置；每个位置只在停车并放下 observe 相机后通过受限 `/cmd_vel` 旋转寻找当前目标类别。完整一轮没有识别到目标时，会围绕同一个已记录失败位置继续重试，不等待新的 `/cube_category`，因此不会逐轮漂离失败区域。局部重搜得到的物块允许位于原随机 A/B/C 区域之外，以覆盖运输途中掉落的情况；物块坐标始终来自实时视觉，不写死物块位置。

如果任务是在获得任何目标 Pose 之前失败（例如 A/B/C 全部因视觉丢帧漏检），执行器会保持同一类别并重新执行正常区域搜索。恢复时若夹爪后端仍报告未知附着，则保持停车并重复安全恢复，不会带着未知附着状态开始搜索。最终成功仍进入 `PARKED -> SUCCESS` 并设置 `/gazebo_success=1`。

## 22. 近距离物块视觉与 C 区 standoff

随机物块靠近观察相机时，亮度连通域可能超过原先 `180 px/15000 px²` 的候选上限，导致日志只出现 `pixel_size` 或 `area` 过滤。当前视觉候选上限调整为 `300 px/30000 px²`，之后仍必须通过形状、纹理、SIFT 类别和深度几何检查，不放宽抓取精度或类别判定。

C 区固定观察点从 `(-1.20, -0.53)` 后移到 `(-1.40, -0.53)`，为随机生成在 C 区近端的物块保留更大相机 standoff；物块随机范围和坐标生成脚本未修改。若近距离问题再次出现，优先查看 `/cube_vision` 的 `pixel_size`、`area`、`depth_plane_invalid` 诊断。

固定区域的中心视角和两个机械臂 pan 视角全部漏检时，执行器现在根据该区域
边界中心计算一个向区域外 `0.18 m` 的临时观察位，再重复同一组视角一次；
该兜底只在漏检时触发，正常命中路径不增加耗时，也不写入随机物块坐标。

另外，固定区域若要求的 yaw 与当前底盘朝向相差超过 `2.50 rad`，会先抬臂
用 `move_base` 原地调整底盘，再使用 nominal observe 姿态，避免用接近关节
限位的 `arm_joint1` 补偿造成中心/近场物块被相机遮挡。

---

## 23. 2026-08-18 随机多轮验证与运输恢复加固

### 23.1 本轮修复

真实 C 区运行中，夹取已进入 `GRASPING/cube_2`，但 transport 抬臂开始时
`r_joint` 仍有短暂运动，触发抓取后端的释放竞态。受约束未修改
`grasp_attach.py`；执行器改为在确认附着并发送 hold 后，使用既有
`settle_timeout/settle_samples/settle_position_delta` 等待 `r_joint` 稳定，再抬臂。
后续成功轮次的 `grasp.hold` 均约为 `0.301-0.351 s`，运输期间未再脱落。

另外两处恢复路径保持小范围修改：

- 停车导航使用正式 `nav_retries`，不再强制 `retries=0`；
- `SEARCH_NEAR_FAILURE` 的受限直接平移超时后，在机械臂已抬起的前提下用
  `move_base` 重试同一局部点，避免持续卡在直接移动超时或旋转漂移边界。

### 23.2 真实 Gazebo 结果

每轮完整任务均先关闭并重新启动 Gazebo，使物块重新随机生成；物块 map
目标仍来自视觉。以下成功轮次最后都满足 `ready=True`、`GRASPING`、
`attached_model` 与类别匹配、车体完全进入对应车间、`/gazebo_success=1`：

```text
类别         随机区域/生成坐标          最终 map 位姿 (x, y, yaw)
daily        B (-1.385,  0.052)          (0.992, -1.500,  0.004)
food         B (-1.484,  0.130)          (0.995, -2.980,  0.005)
electronics  B (-1.484,  0.013)          (2.540, -2.220,  0.004)
food         A (-2.038, -0.483)          (0.989, -2.980, -0.006)
electronics  C (-0.845, -0.392)          (2.549, -2.220,  0.005)
food         C (-0.930, -0.534)          (0.995, -2.980, -0.004)
```

A 区 food 的视觉目标为 `(-1.987, -0.505)`。C 区 electronics 首次
`PREPARE_GRASP` 被保守抓取框拒绝，offset 重对位后成功；视觉目标从
`(-0.774, -0.372)` 修正到 `(-0.776, -0.415)`。靠 C 区相机侧边缘的
food 生成于 `(-0.930, -0.534)`，由视觉估计为 `(-0.861, -0.543)` 并完成
抓取和停车。这些生成坐标只来自仿真启动日志，用于记录随机覆盖，未写入代码。

无效类别 `invalid` 的单独验证保持 `WAITING_FOR_CATEGORY`，且
`/gazebo_success=0`。

### 23.3 大 yaw 与 standoff 当前结论

一次 electronics/C 随机运行已真实出现：

```text
yaw delta 3.017 exceeds 2.500; reorienting base with move_base
base reoriented; using nominal observe joint1
missed from nominal view; retrying 0.18 m farther ... (-1.580, -0.532)
```

大 yaw 重定向和 standoff 移动本身执行成功，但该次后退视角没有重新识别
目标；目标由第二轮正常 C 区视角找到。另一次 C 区抓取运输保持附着，但停车
两种朝向均失败，随后局部恢复卡在直接移动超时/`0.080 m` 旋转漂移边界；
这两次失败分别促成 23.1 的 settle、停车重试和局部 move_base 回退。

本次新增的 C 区及 C 边缘成功轮次都在粗旋转 `3.41 rad` 时提前找到目标，
没有进入固定区域 standoff 分支。因此目前可以确认大 yaw/standoff 日志和动作
已真实触发，也可以确认 C 区中心及边缘物块能完成任务，但仍不能宣称
“后退 0.18 m 后重新识别并继续成功”已经闭环。后续随机轮次仍需专门捕获该
组合条件，不应通过写死坐标或修改生成脚本制造样本。

### 23.4 检查

修改后已通过：

```text
python3 -m py_compile src/car3/scripts/pick_place_executor.py
3 个视觉单元测试
catkin_make --pkg car3 -j2
git diff --check
```

## 24. 2026-08-18 第一版快速区域搜索复测

第一版快速路径已保留为可配置实验项 `fast_area_search`。它从一次粗搜索位姿
依次使用 A/B/C 的中心和两个 `arm_joint1` pan 视角；目标候选仍需回到该区域
标准观察点独立复核，区域外视觉 Pose 会被拒绝，完整粗旋转和固定区域多轮搜索
仍是兜底。快速探测阶段不会触发 standoff，避免候选失败后重复后退观察。

重新启动仿真取得随机分布后，实测：

```text
electronics / B: 初次快速路径 fast_area.total=139.475 s；当时还会在快速阶段
                 触发 standoff，随后已加 guard 避免重复，最终 SUCCESS
electronics / C: fast_area.total=105.477 s，标准 C 视角复核成功，最终 SUCCESS
electronics / A: 正式粗旋转优先，search.total=80.184 s，固定 A 视角成功，
                  首次 PREPARE_GRASP 失败后重对位，GRASPING/cube_2 并 SUCCESS
```

B/C 快速路径明显慢于完整粗旋转，因此正式配置将 `fast_area_search` 设为
`false`；代码和参数仍保留，后续只有在真实视场或硬件动作时间改变并重新测量
后才可重新启用。B 区标准观察点同步后移到 `(-1.40, -0.58)`，该位置来自
区域中心向外的既有 `0.18 m` standoff 规则，不包含任何随机物块坐标。

## 25. 2026-08-18 RECOVER_FAILURE 有界恢复

一次 food/A 运行已成功进入 `GRASPING/cube_0`，随后 transport 机械臂未能在
既有容差内稳定。夹爪释放后，恢复阶段重复发送与 transport 相同的 navigation
姿态，但每次都出现 `navigation arm pose did not settle`；原主循环没有失败上限，
因此会永久重复 `RECOVER_FAILURE`。

执行器现在保留原安全约束和恢复动作，但连续三次恢复失败后会：

```text
发布零速度
状态设为 FAILED
结果发布 FAILED:failure recovery did not complete after 3 attempts
保持 /gazebo_success=0
退出任务执行器
```

任何一次恢复成功都会清零失败计数，并继续当前类别的失败点局部重搜。机械臂
settle 超时日志现在包含五个关节各自的 target、actual 和 error，便于定位实际
未收敛关节；没有放宽关节容差、稳定样本或安全姿态。

新增 `test_pick_place_recovery.py` 模拟恢复永久失败，确认只调用三次恢复并进入
`FAILED`。随机 Gazebo 回归中 food 位于 A 区 `(-2.064, -0.504)`，任务正常完成
`ready=True`、`GRASPING/cube_0`、transport、停车和 `SUCCESS`，正常路径未回归。

## 26. 2026-08-18 粗旋转停车确认与视觉频率

视觉处理频率从 `5 Hz` 提高到保守的 `8 Hz`；Gazebo RGB 和 depth 相机均为
`20 Hz`，因此仍保留一半以上的计算余量。`stability_samples=5`、
`required_votes=3`、SIFT 参数和旋转速度均未放宽。生产配置关闭调试图；RGB
和 depth 调试图现在只有在开关启用且存在订阅者时才生成，避免无用的图像复制
与序列化。

粗旋转看到目标类别后不再直接返回运动中的视觉 Pose。执行器会立即发布零速度，
等待底盘稳定，reset 视觉历史，并要求新的静止稳定 Pose；静止 Pose 与运动候选
的 map 平面距离还必须不超过既有 `search_target_confirm_distance=0.05 m`。
确认失败时更新视觉序列屏障，从当前底盘 yaw 继续完成剩余旋转，而不是接受旧帧
或直接跳过 A/B/C 固定视角兜底。

新增 `test_pick_place_search.py` 覆盖静止近邻 Pose 接受和偏移 Pose 拒绝。当前检查：

```text
python3 -m unittest ...: 6 tests, OK
catkin_make --pkg car3 -j2: passed
catkin_make run_tests_car3 -j2: 6 tests, 0 errors, 0 failures
python3 -m py_compile: passed
git diff --check: passed
```

新的随机 Gazebo 实例确认参数服务器使用
`/cube_vision/processing_rate=8.0`、`publish_debug_image=False`。完成回归的随机
分布为 electronics/C `(-0.845, -0.607)`、daily/A `(-1.976, -0.453)`、
food/B `(-1.481, 0.124)`，发布任务为 `electronics`。粗搜索先确认 daily 和 food
非目标，然后在旋转 `3.54 rad` 时发现 electronics，停车后的独立静止确认日志为：

```text
stopped target confirmed: category=electronics confidence=1.000 shift=0.018 m
search.coarse.stationary_confirm: 1.703 s accepted=True
search.coarse.rotation: 17.797 s
search.total: 52.079 s source=coarse
```

静止视觉目标为 `(-0.816, -0.606)`，分类到 `area_c` 后完成对位。抓取准备得到
`ready=True`，夹取后为 `GRASPING/cube_2`；运输和停车完成后状态为 `SUCCESS`。
任务结束后独立读取确认 `/gazebo_success=1`、`ready=True`、
`grasp_state=GRASPING`、`attached_model=cube_2`。本轮证明新增停车确认不会破坏
C 区近场边缘目标的完整抓取运输流程；单轮结果仍不能替代 food/daily 以及 A/B
区域的后续随机多轮覆盖。

## 27. 2026-08-18 靠墙物块两段式抓取

为避免机械臂从收拢姿态直接下降时向墙面扫动，抓取准备改为两段轨迹：先到
`grasp_approach`，再下降到原 `grasp`。新 approach 的 TCP 约在最终抓取点后方
`0.045 m`、上方 `0.11 m`；最终抓取位姿和视觉对位目标均未改变。激光平面高于
物块，本改动不使用雷达寻找物块或估计物块位置。

干净重启 Gazebo 后，daily/cube_1 随机生成在 C 区
`(-0.799, -0.581)`，距该区域东侧边界约 `0.029 m`。生成坐标只用于记录随机
覆盖，正式任务仍使用视觉 Pose。任务在固定 C 区视角确认目标，并真实触发：

```text
yaw delta 3.048 exceeds 2.500; reorienting base with move_base
base reoriented; using nominal observe joint1
timing grasp.prepare.approach_arm: 3.966 s
timing grasp.prepare.descend_arm: 3.513 s
ready=True offset=(0.0044, -0.0048, 0.0107)
grasp_state=GRASPING attached_model=cube_1
```

随后车辆保持抓取完成 daily 停车，状态为 `SUCCESS`。独立读取确认
`/gazebo_success=1`、`/grasp_attach/ready=True`、
`/grasp_attach/state=GRASPING`、`/grasp_attach/attached_model=cube_1`。
这一轮确认两段式下降可完成靠边 daily 的抓取与运输；仍需按既有要求继续用每轮
重新随机启动的方式覆盖其他墙边朝向，不能用这一轮替代三类别多轮回归。

新增单元测试确认 `_prepare_stationary_grasp()` 严格按
`grasp_approach -> grasp` 执行。当前检查为 `catkin_make --pkg car3 -j2` 通过，
`run_tests_car3` 共 7 个测试、0 错误、0 失败，`git diff --check` 通过。

## 28. 2026-08-18 近距离 A/B/C 顺序区域搜索

粗旋转漏检后的第一轮区域搜索现在严格先完成 `area_a -> area_b -> area_c`，
每个区域使用中心、`+0.45 rad`、`-0.45 rad` 三个机械臂视角。第一轮不再在 A
漏检后立即执行 A 的 standoff；只有完整 A/B/C 均漏检，第二轮才允许原有
`0.18 m` 后退兜底。目标一经确认仍立即结束搜索。

三个固定观察点改为位于各自区域中心外侧约 `0.45 m`：

```text
area_a: (-1.560, -0.445)，从东侧观察
area_b: (-1.395, -0.370)，从南侧观察
area_c: (-1.310, -0.525)，从西侧观察
```

区域观察不再复用粗搜索的伸展 `observe` 姿态。新增 `area_observe` 将相机收回并
抬高，理论地面视线中心约在底盘前方 `0.37 m`；随后使用 `arm_joint1` 左右 pan
覆盖区域横向边缘。粗旋转仍使用原 `observe`，抓取和运输姿态未改变。

真实 Gazebo 确定性分支验证仅将该测试进程的粗旋转角临时设为 `0.01 rad`，
正式 YAML 已恢复 `6.283185`。本轮随机分布为 food/A
`(-2.029, -0.601)`、daily/B `(-1.423, 0.027)`、electronics/C
`(-0.943, -0.453)`；发布 `electronics` 后：

```text
coarse search missed; visiting area_a -> area_b -> area_c ...
A 中心和 +0.45 视角确认 food 非目标
B 中心和 -0.45 视角确认 daily 非目标
C 中心视角分类冲突后，+0.45 视角以 3/3 确认 electronics
search.total: source=area_c pass=1
```

三个近距离 waypoint 均由直接移动安全到达，第一轮没有 standoff 插入 A/B/C
顺序。随后任务完成 `ready=True`、`GRASPING/cube_2`、停车 `SUCCESS`；独立
读取确认 `/gazebo_success=1`、`attached_model=cube_2`。随机生成坐标只用于
记录覆盖，物块定位仍来自视觉。

## 29. 2026-08-18 搜索视野停车确认

为减少运动中视觉误分类造成的无效搜索，粗旋转期间一旦出现任何已定位物块，
执行器都会立即停车并执行独立静止确认。确认失败时清除本轮视觉屏障后从当前
yaw 继续搜索；确认是非目标时抬臂退出粗旋转，转入固定的 `area_a -> area_b ->
area_c` 顺序，不再继续旋转整圈。

粗观察姿态已经得到有效物块位姿时，也先执行同样的静止确认，不会因为它是非
目标而直接进入旋转。

固定区域中心视角确认到非目标后默认立即结束该区域并进入下一区域；只有没有
稳定物块确认时，才继续该区域的左右机械臂 pan 视角和第二轮 standoff 兜底。
固定区域的观察 yaw 由 waypoint 指向对应区域边界中心自动计算，YAML 中的 yaw
仅作为缺少区域边界时的后备值；短距离 A/B/C 转移保持当前底盘 yaw，优先用
`arm_joint1` 补偿，只有机械臂没有合法补偿时才允许 yaw 重定向。观察姿态按
`observe` 远距离中心视角、`area_observe` 近距离中心及左右 pan 视角分阶段执行。
新增单元测试覆盖粗旋转发现非目标后停车确认并退出粗搜索。该行为已通过
`catkin_make --pkg car3 -j2`、7 个 Python 搜索/恢复测试和 `git diff --check`；
需要在干净随机 Gazebo 重启中继续覆盖三种类别和三个区域。
