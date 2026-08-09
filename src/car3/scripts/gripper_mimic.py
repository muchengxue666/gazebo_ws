#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState

MIMIC_JOINTS = {
    'l_joint':     -1.0,
    'r_in_joint':   1.0,
    'l_in_joint':  -1.0,
    'r_out_joint': -1.0,
    'l_out_joint':  1.0,
}

MIMIC_NAMES = list(MIMIC_JOINTS.keys())

class GripperJointStateAugmenter:
    def __init__(self):
        self.sub = rospy.Subscriber('/joint_states', JointState, self.callback)
        self.pub = rospy.Publisher('/joint_states_full', JointState, queue_size=10)
        rospy.loginfo('夹爪关节增强节点已启动 — 发布 /joint_states_full')

    def callback(self, msg: JointState):
        if 'r_joint' not in msg.name:
            rospy.logwarn_throttle(5, 'joint_states 中未找到 r_joint，等待中...')
            self.pub.publish(msg)
            return

        r_idx = msg.name.index('r_joint')
        r_pos = msg.position[r_idx]
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = list(msg.position)
        out.velocity = list(msg.velocity) if msg.velocity else []
        out.effort = list(msg.effort) if msg.effort else []
        for mimic_name, ratio in MIMIC_JOINTS.items():
            if mimic_name not in out.name:
                mimic_pos = r_pos * ratio
                out.name.append(mimic_name)
                out.position.append(mimic_pos)
                if out.velocity:
                    r_vel = msg.velocity[r_idx] if len(msg.velocity) > r_idx else 0.0
                    out.velocity.append(r_vel * abs(ratio))
                if out.effort:
                    r_eff = msg.effort[r_idx] if len(msg.effort) > r_idx else 0.0
                    out.effort.append(r_eff * abs(ratio))

        self.pub.publish(out)


if __name__ == '__main__':
    rospy.init_node('gripper_mimic_node')
    node = GripperJointStateAugmenter()
    rospy.spin()
