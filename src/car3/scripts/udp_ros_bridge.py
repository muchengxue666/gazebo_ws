#!/usr/bin/env python3
"""Bridge /cube_category and /gazebo_success over UDP JSON datagrams.

Each datagram contains one JSON object. JSON Lines in one datagram are also
accepted. Messages use the same fields as tcp_ros_bridge.py:

  {"type": "cube_category", "topic": "/cube_category", "value": "food"}
  {"type": "gazebo_success", "param": "/gazebo_success", "value": 1}

Incoming messages update the corresponding ROS topic/parameter. Local topic
and parameter changes are sent to ``~remote_host:~remote_port``.
"""

import json
import queue
import socket
import threading
import time

import rospy
from std_msgs.msg import String


class UdpRosBridge:
    def __init__(self):
        self.bind_host = rospy.get_param('~bind_host', '0.0.0.0')
        self.bind_port = int(rospy.get_param('~bind_port', 9000))
        self.remote_host = rospy.get_param('~remote_host', '192.168.10.246')
        self.remote_port = int(rospy.get_param('~remote_port', 9000))
        self.param_poll_period = float(
            rospy.get_param('~param_poll_period', 0.2))
        self.category_topic = rospy.get_param(
            '~category_topic', '/cube_category')
        self.success_param = rospy.get_param(
            '~success_param', '/gazebo_success')

        if not self.bind_host:
            raise rospy.ROSInitException('bind_host is empty')
        if not self.remote_host:
            raise rospy.ROSInitException('remote_host is empty')
        if not 1 <= self.bind_port <= 65535:
            raise rospy.ROSInitException('bind_port is outside 1..65535')
        if not 1 <= self.remote_port <= 65535:
            raise rospy.ROSInitException('remote_port is outside 1..65535')
        if self.param_poll_period <= 0.0:
            raise rospy.ROSInitException('param_poll_period must be positive')

        self.category_pub = rospy.Publisher(
            self.category_topic, String, queue_size=1)
        rospy.Subscriber(
            self.category_topic, String, self._category_cb, queue_size=1)
        self._outgoing = queue.Queue(maxsize=100)
        self._suppress_category = None
        self._last_success = object()
        self._thread = None

    def _message(self, message_type, value):
        message = {'type': message_type, 'value': value}
        if message_type == 'cube_category':
            message['topic'] = self.category_topic
        else:
            message['param'] = self.success_param
        return message

    def _enqueue(self, message):
        try:
            self._outgoing.put_nowait(message)
        except queue.Full:
            rospy.logwarn_throttle(5.0, 'UDP bridge outgoing queue is full')

    def _category_cb(self, msg):
        value = msg.data.strip()
        if not value:
            return
        suppressed = self._suppress_category
        if (suppressed is not None and suppressed[0] == value
                and time.monotonic() <= suppressed[1]):
            self._suppress_category = None
            return
        self._enqueue(self._message('cube_category', value))

    def _poll_success_param(self):
        value = rospy.get_param(self.success_param, 0)
        if isinstance(value, bool):
            normalized = int(value)
        elif isinstance(value, (int, float)):
            normalized = int(value)
        else:
            normalized = str(value)
        if normalized != self._last_success:
            self._last_success = normalized
            self._enqueue(self._message('gazebo_success', normalized))

    def _handle_message(self, message):
        if not isinstance(message, dict):
            return
        message_type = message.get('type') or message.get('topic') \
            or message.get('param')
        value = message.get('value', message.get('data'))
        if message_type in ('cube_category', self.category_topic):
            if value is None:
                return
            value = str(value).strip()
            if not value:
                return
            self._suppress_category = (value, time.monotonic() + 1.0)
            self.category_pub.publish(String(data=value))
        elif message_type in ('gazebo_success', self.success_param):
            if value is None:
                return
            rospy.set_param(self.success_param, value)
            self._last_success = value

    def _send_pending(self, sock):
        destination = (self.remote_host, self.remote_port)
        while True:
            try:
                message = self._outgoing.get_nowait()
            except queue.Empty:
                return
            payload = json.dumps(
                message, separators=(',', ':')).encode('utf-8')
            sock.sendto(payload, destination)

    def _handle_datagram(self, payload, address):
        for raw in payload.splitlines() or [payload]:
            if not raw.strip():
                continue
            try:
                self._handle_message(json.loads(raw.decode('utf-8')))
            except (UnicodeDecodeError, json.JSONDecodeError,
                    TypeError, ValueError) as exc:
                rospy.logwarn(
                    'ignoring malformed UDP JSON from %s:%d: %s',
                    address[0], address[1], exc)

    def _network_loop(self, sock):
        try:
            rospy.loginfo(
                'UDP bridge listening on %s:%d, sending to %s:%d',
                self.bind_host, self.bind_port,
                self.remote_host, self.remote_port)
            while not rospy.is_shutdown():
                try:
                    self._send_pending(sock)
                    payload, address = sock.recvfrom(65535)
                    self._handle_datagram(payload, address)
                except socket.timeout:
                    continue
                except OSError as exc:
                    rospy.logwarn_throttle(5.0, 'UDP bridge socket error: %s', exc)
        finally:
            sock.close()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.bind_host, self.bind_port))
            sock.settimeout(0.2)
        except OSError as exc:
            sock.close()
            raise rospy.ROSInitException(
                'cannot bind UDP bridge to {}:{}: {}'.format(
                    self.bind_host, self.bind_port, exc))
        self._thread = threading.Thread(
            target=self._network_loop, args=(sock,),
            name='udp_ros_bridge', daemon=True)
        self._thread.start()
        rate = rospy.Rate(1.0 / self.param_poll_period)
        while not rospy.is_shutdown():
            self._poll_success_param()
            rate.sleep()


def main():
    rospy.init_node('udp_ros_bridge')
    try:
        UdpRosBridge().run()
    except rospy.ROSInitException as exc:
        rospy.logfatal('UDP ROS bridge configuration error: %s', exc)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
