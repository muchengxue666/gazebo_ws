#!/usr/bin/env python3
import math
import time

import actionlib
import rospy
import tf.transformations as transformations
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def main():
    rospy.init_node('test_move_base_goal')

    x = float(rospy.get_param('~x', -2.36))
    y = float(rospy.get_param('~y', -0.445))
    yaw = float(rospy.get_param('~yaw', 0.0))
    startup_delay = float(rospy.get_param('~startup_delay', 10.0))
    timeout = float(rospy.get_param('~timeout', 60.0))

    client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    rospy.loginfo('waiting for /move_base action server')
    if not client.wait_for_server(rospy.Duration(60.0)):
        rospy.logerr('/move_base action server unavailable')
        return

    rospy.loginfo('waiting %.1f seconds before sending goal', startup_delay)
    time.sleep(startup_delay)

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time(0)
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    quaternion = transformations.quaternion_from_euler(0.0, 0.0, yaw)
    goal.target_pose.pose.orientation.x = quaternion[0]
    goal.target_pose.pose.orientation.y = quaternion[1]
    goal.target_pose.pose.orientation.z = quaternion[2]
    goal.target_pose.pose.orientation.w = quaternion[3]

    rospy.loginfo('sending goal: x=%.3f y=%.3f yaw=%.3f', x, y, yaw)
    client.send_goal(goal)

    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo('navigation succeeded')
            return
        if state in (GoalStatus.ABORTED, GoalStatus.REJECTED,
                     GoalStatus.PREEMPTED, GoalStatus.RECALLED,
                     GoalStatus.LOST):
            rospy.logerr(
                'navigation failed: state=%d text=%s',
                state, client.get_goal_status_text())
            return
        time.sleep(0.1)

    client.cancel_goal()
    rospy.logerr('navigation timed out after %.1f seconds', timeout)


if __name__ == '__main__':
    main()
