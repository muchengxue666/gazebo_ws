#!/usr/bin/env python3
import rospy
import tf.transformations as T
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64, Bool, String
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from gazebo_msgs.srv import GetLinkState, GetModelState, GetWorldProperties, SetModelState
from gazebo_msgs.msg import ModelState


def _qv_mult(q, v):
    qv = T.quaternion_multiply(
        T.quaternion_multiply(q, [v[0], v[1], v[2], 0.0]),
        T.quaternion_conjugate(q))
    return (qv[0], qv[1], qv[2])


class GraspAttach:
    def __init__(self):
        self.close_threshold = rospy.get_param('~gripper_close_threshold', 0.8)
        self.open_threshold = rospy.get_param('~gripper_open_threshold', 0.89)
        self.gripper_link = rospy.get_param('~gripper_link', 'car3::tcp_link')
        self.object_models = rospy.get_param('~object_models', ['cube_0', 'cube_1', 'cube_2'])
        self.current_object = None  
        self.update_rate = rospy.get_param('~update_rate', 100)
        self.check_rate = rospy.get_param('~check_rate', 2)
        self.obj_half_x = rospy.get_param('~object_half_x', 0.02)
        self.obj_half_y = rospy.get_param('~object_half_y', 0.02)
        self.obj_half_z = rospy.get_param('~object_half_z', 0.02)

        self.state = 'IDLE'
        self.r_joint_pos = None
        self.offset_pos = None
        self.offset_quat = None
        self.grasp_success = False

        rospy.loginfo('等待 Gazebo 服务...')
        rospy.wait_for_service('/gazebo/get_link_state')
        rospy.wait_for_service('/gazebo/get_model_state')
        rospy.wait_for_service('/gazebo/get_world_properties')
        rospy.wait_for_service('/gazebo/set_model_state')
        self.get_link_srv = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        self.get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.get_world_srv = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        self.set_model_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._model_list = []
        self._model_list_stamp = rospy.Time(0) 
        rospy.loginfo('Gazebo 服务已就绪')

        self.dist_pub = rospy.Publisher('~distance', Float32, queue_size=5)
        self.ready_pub = rospy.Publisher('~ready', Bool, queue_size=5)
        self.offset_pub = rospy.Publisher('~offset', Point, queue_size=5)
        self.state_pub = rospy.Publisher('~state', String, queue_size=5)
        self.attached_model_pub = rospy.Publisher(
            '~attached_model', String, queue_size=1, latch=True)
        self.attached_model_pub.publish(String(data=''))

        self.gripper_pub = rospy.Publisher(
            '/gripper_controller/command', Float64, queue_size=1)

        self.joint_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_cb)

        self._check_timer = rospy.Timer(
            rospy.Duration(1.0 / self.check_rate), self._check_cb)

        self._follow_timer = None

        rospy.loginfo(f'grasp_attach 就绪 '
                      f'(close<{self.close_threshold}, open>{self.open_threshold}, '
                      f'obj=({self.obj_half_x},{self.obj_half_y},{self.obj_half_z}), '
                      f'models={self.object_models}, '
                      f'check={self.check_rate}Hz, follow={self.update_rate}Hz)')

    def _joint_cb(self, msg):
        if 'r_joint' not in msg.name:
            return
        self.r_joint_pos = msg.position[msg.name.index('r_joint')]
        self._tick_state()

    def _tick_state(self):
        if self.r_joint_pos is None:
            return
        if self.state == 'IDLE' and self.r_joint_pos < self.close_threshold:
            self._do_grasp()
        elif self.state == 'GRASPING' and self.r_joint_pos > self.open_threshold:
            self._do_release()

    def _get_gripper_pose(self):
        try:
            gripper = self.get_link_srv(self.gripper_link, 'world')
            if not gripper.success:
                return None
            return gripper.link_state.pose
        except rospy.ServiceException:
            return None

    def _refresh_model_list(self):
        now = rospy.Time.now()
        if (now - self._model_list_stamp).to_sec() < 2.0:
            return
        try:
            resp = self.get_world_srv()
            self._model_list = resp.model_names
            self._model_list_stamp = now
        except rospy.ServiceException:
            pass

    def _get_model_offset(self, model_name, gripper_pose=None):
        self._refresh_model_list()
        if model_name not in self._model_list:
            return None

        if gripper_pose is None:
            gripper_pose = self._get_gripper_pose()
            if gripper_pose is None:
                return None
        g_pos = gripper_pose.position
        g_q = [gripper_pose.orientation.x, gripper_pose.orientation.y,
               gripper_pose.orientation.z, gripper_pose.orientation.w]
        try:
            obj = self.get_model_srv(model_name, 'world')
            if not obj.success:
                return None
            o_pos = obj.pose.position
            o_q = [obj.pose.orientation.x, obj.pose.orientation.y,
                   obj.pose.orientation.z, obj.pose.orientation.w]
        except rospy.ServiceException:
            return None
        world_dp = (o_pos.x - g_pos.x, o_pos.y - g_pos.y, o_pos.z - g_pos.z)
        g_q_inv = T.quaternion_inverse(g_q)
        local = _qv_mult(g_q_inv, world_dp)
        dist = (world_dp[0]**2 + world_dp[1]**2 + world_dp[2]**2) ** 0.5
        return (local, dist, o_q)

    def _find_closest_in_box(self):
        gripper_pose = self._get_gripper_pose()
        if gripper_pose is None:
            return None
        best = None
        best_dist = float('inf')
        for model_name in self.object_models:
            result = self._get_model_offset(model_name, gripper_pose)
            if result is None:
                continue
            (px, py, pz), dist, o_q = result
            in_box = (abs(px) <= self.obj_half_x and
                      abs(py) <= self.obj_half_y and
                      abs(pz) <= self.obj_half_z)
            if in_box and dist < best_dist:
                best_dist = dist
                best = (model_name, px, py, pz, dist, o_q)
        return best

    def _check_cb(self, event):
        best_dist = float('inf')
        best_px = best_py = best_pz = 0.0
        best_in_box = False
        for model_name in self.object_models:
            result = self._get_model_offset(model_name)
            if result is None:
                continue
            (px, py, pz), dist, _ = result
            if dist < best_dist:
                best_dist = dist
                best_px, best_py, best_pz = px, py, pz
                best_in_box = (abs(px) <= self.obj_half_x and
                               abs(py) <= self.obj_half_y and
                               abs(pz) <= self.obj_half_z)
        self.dist_pub.publish(Float32(data=best_dist))
        self.ready_pub.publish(Bool(data=best_in_box))
        self.offset_pub.publish(Point(x=best_px, y=best_py, z=best_pz))
        self.state_pub.publish(String(data=self.state))

    def _do_grasp(self):
        best = self._find_closest_in_box()
        if best is None:
            return  
        model_name, px, py, pz, dist, o_q = best

        self.current_object = model_name
        self.offset_pos = (px, py, pz)

        gripper_pose = self._get_gripper_pose()
        g_q = [gripper_pose.orientation.x, gripper_pose.orientation.y,
               gripper_pose.orientation.z, gripper_pose.orientation.w]
        g_q_inv = T.quaternion_inverse(g_q)
        self.offset_quat = T.quaternion_multiply(g_q_inv, o_q)

        self.grasp_success = True
        self.state = 'GRASPING'
        self.attached_model_pub.publish(String(data=self.current_object))

        if self.r_joint_pos is not None:
            self.gripper_pub.publish(Float64(data=self.r_joint_pos))

        if self._follow_timer is None:
            self._follow_timer = rospy.Timer(
                rospy.Duration(1.0 / self.update_rate), self._follow_cb)

    def _do_release(self):
        if self._follow_timer is not None:
            self._follow_timer.shutdown()
            self._follow_timer = None
        self.offset_pos = None
        self.offset_quat = None
        self.current_object = None
        self.grasp_success = False
        self.state = 'IDLE'
        self.attached_model_pub.publish(String(data=''))

    def _follow_cb(self, event):
        if self.offset_pos is None or self.current_object is None:
            return
        try:
            gripper = self.get_link_srv(self.gripper_link, 'world')
            if not gripper.success:
                return
            g_pos = gripper.link_state.pose.position
            g_q = [gripper.link_state.pose.orientation.x,
                   gripper.link_state.pose.orientation.y,
                   gripper.link_state.pose.orientation.z,
                   gripper.link_state.pose.orientation.w]
        except rospy.ServiceException:
            return

        rotated = _qv_mult(g_q, self.offset_pos)
        target_pos = Point(x=g_pos.x + rotated[0],
                           y=g_pos.y + rotated[1],
                           z=g_pos.z + rotated[2])

        q_w = T.quaternion_multiply(g_q, self.offset_quat)
        target_quat = Quaternion(x=q_w[0], y=q_w[1], z=q_w[2], w=q_w[3])

        st = ModelState()
        st.model_name = self.current_object
        st.pose = Pose(position=target_pos, orientation=target_quat)
        st.twist = Twist()
        st.reference_frame = 'world'

        try:
            result = self.set_model_srv(st)
            if not result.success:
                rospy.logerr_throttle(
                    1.0, 'Failed to keep %s attached: %s',
                    self.current_object, result.status_message)
        except rospy.ServiceException as exc:
            rospy.logerr_throttle(
                1.0, 'Failed to update attached object: %s', exc)


if __name__ == '__main__':
    rospy.init_node('grasp_attach')
    node = GraspAttach()
    rospy.spin()
