#!/usr/bin/env python3
"""Bridge /cube_category and /gazebo_success over a TCP JSONL connection.

Each line is a JSON object.  Messages sent by this node are:

  {"type": "cube_category", "topic": "/cube_category", "value": "food"}
  {"type": "gazebo_success", "param": "/gazebo_success", "value": 1}

The connection is a client connection to ``~remote_host:~remote_port`` and is
re-established after disconnects.  Incoming messages update the corresponding
ROS topic/parameter; outgoing messages are generated from local changes.
"""

import json
import queue
import socket
import threading
import time

import rospy
from std_msgs.msg import String


class TcpRosBridge:
    def __init__(self):
        self.remote_host = rospy.get_param('~remote_host', '192.168.10.246')
        self.remote_port = int(rospy.get_param('~remote_port', 9000))
        self.connect_timeout = float(
            rospy.get_param('~connect_timeout', 3.0))
        self.reconnect_delay = float(
            rospy.get_param('~reconnect_delay', 2.0))
        self.param_poll_period = float(
            rospy.get_param('~param_poll_period', 0.2))
        self.category_topic = rospy.get_param(
            '~category_topic', '/cube_category')
        self.success_param = rospy.get_param(
            '~success_param', '/gazebo_success')

        if not self.remote_host:
            raise rospy.ROSInitException('remote_host is empty')
        if not 1 <= self.remote_port <= 65535:
            raise rospy.ROSInitException('remote_port is outside 1..65535')
        if self.connect_timeout <= 0.0 or self.reconnect_delay <= 0.0:
            raise rospy.ROSInitException(
                'connect_timeout and reconnect_delay must be positive')
        if self.param_poll_period <= 0.0:
            raise rospy.ROSInitException('param_poll_period must be positive')

        self.category_pub = rospy.Publisher(
            self.category_topic, String, queue_size=1)
        rospy.Subscriber(
            self.category_topic, String, self._category_cb, queue_size=1)
        self._outgoing = queue.Queue(maxsize=100)
        self._send_lock = threading.Lock()
        self._suppress_category = None
        self._last_success = object()
        self._thread = threading.Thread(
            target=self._network_loop, name='tcp_ros_bridge', daemon=True)

    @staticmethod
    def _message(message_type, value):
        return {
            'type': message_type,
            'value': value,
            'topic': '/cube_category' if message_type == 'cube_category'
            else None,
            'param': '/gazebo_success' if message_type == 'gazebo_success'
            else None,
        }

    def _enqueue(self, message):
        try:
            self._outgoing.put_nowait(message)
        except queue.Full:
            rospy.logwarn_throttle(5.0, 'TCP bridge outgoing queue is full')

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
        message_type = message.get('type') or message.get('topic')
        value = message.get('value', message.get('data'))
        if message_type in ('cube_category', '/cube_category'):
            if value is None:
                return
            value = str(value).strip()
            if not value:
                return
            self._suppress_category = (value, time.monotonic() + 1.0)
            self.category_pub.publish(String(data=value))
        elif message_type in ('gazebo_success', '/gazebo_success'):
            if value is None:
                return
            rospy.set_param(self.success_param, value)
            self._last_success = value

    def _send_pending(self, sock):
        while True:
            try:
                message = self._outgoing.get_nowait()
            except queue.Empty:
                return
            payload = (json.dumps(message, separators=(',', ':')) + '\n').encode(
                'utf-8')
            with self._send_lock:
                sock.sendall(payload)

    def _network_loop(self):
        while not rospy.is_shutdown():
            sock = None
            try:
                rospy.loginfo(
                    'connecting TCP bridge to %s:%d',
                    self.remote_host, self.remote_port)
                sock = socket.create_connection(
                    (self.remote_host, self.remote_port),
                    timeout=self.connect_timeout)
                sock.settimeout(0.2)
                rospy.loginfo('TCP bridge connected')
                buffer = b''
                while not rospy.is_shutdown():
                    self._send_pending(sock)
                    try:
                        chunk = sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        raise ConnectionError('remote closed TCP connection')
                    buffer += chunk
                    while b'\n' in buffer:
                        raw, buffer = buffer.split(b'\n', 1)
                        if not raw.strip():
                            continue
                        try:
                            self._handle_message(json.loads(raw.decode('utf-8')))
                        except (UnicodeDecodeError, json.JSONDecodeError,
                                TypeError, ValueError) as exc:
                            rospy.logwarn('ignoring malformed TCP JSON: %s', exc)
            except (OSError, ConnectionError) as exc:
                rospy.logwarn_throttle(5.0, 'TCP bridge disconnected: %s', exc)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            time.sleep(self.reconnect_delay)

    def run(self):
        self._thread.start()
        rate = rospy.Rate(1.0 / self.param_poll_period)
        while not rospy.is_shutdown():
            self._poll_success_param()
            rate.sleep()


def main():
    rospy.init_node('tcp_ros_bridge')
    try:
        TcpRosBridge().run()
    except rospy.ROSInitException as exc:
        rospy.logfatal('TCP ROS bridge configuration error: %s', exc)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
