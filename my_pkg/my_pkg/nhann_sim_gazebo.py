#!/usr/bin/env python3
"""
Subscriber phía xe/mô phỏng.
- Giữ nguyên thông số động học từ nhann.py.
- # các phần GPIO/PWM của Raspberry Pi 4.
- Thay bằng publish sang Gazebo mecanum_drive_controller/cmd_vel.
- Bổ sung slew limiter nhẹ để tránh lật xe trong Gazebo khi publisher đổi lệnh quá gắt.
- Bản entry-fix này chỉ siết nhẹ mặc định gia tốc quay để hợp với publisher có ENTRY GUARD.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def slew(current: float, target: float, max_delta: float) -> float:
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


class OmniRobotSimSub(Node):
    def __init__(self):
        super().__init__('omni_robot_sim_sub_left_memory_safe_entryfix')

        alpha = np.array([[43.88 * np.pi / 180],
                          [133.88 * np.pi / 180],
                          [-133.88 * np.pi / 180],
                          [-43.88 * np.pi / 180]])
        beta = np.array([[-46.12 * np.pi / 180],
                         [46.12 * np.pi / 180],
                         [-46.12 * np.pi / 180],
                         [46.12 * np.pi / 180]])
        gamma = np.array([[np.pi / 4],
                          [-np.pi / 4],
                          [np.pi / 4],
                          [-np.pi / 4]])
        l = ((0.13 ** 2 + 0.125 ** 2) ** 0.5) / 2

        J2 = np.array([
            [0.025, 0, 0, 0],
            [0, 0.025, 0, 0],
            [0, 0, 0.025, 0],
            [0, 0, 0, 0.025],
        ])
        J1 = np.array([
            np.sin(alpha + beta + gamma),
            -np.cos(alpha + beta + gamma),
            -l * np.cos(beta + gamma)
        ]).T * (1 / np.cos(np.pi / 4))

        self.J2_inv = np.linalg.inv(J2)
        self.J1 = J1

        self.declare_parameter('topic', '/cmd_vel')
        self.declare_parameter('max_vx', 0.5)
        self.declare_parameter('max_vy', 0.5)
        self.declare_parameter('max_wz', 1.5)
        self.declare_parameter('pwm_div', 40.0)
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('timeout', 0.3)
        self.declare_parameter('gain1', 1.0)
        self.declare_parameter('gain2', 1.3)
        self.declare_parameter('gain3', 1.3)
        self.declare_parameter('gain4', 1.0)

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('lidar_model', 'RPLIDAR C1M1')
        self.declare_parameter('lidar_frame', 'laser_frame')
        self.declare_parameter('lidar_min_range', 0.05)
        self.declare_parameter('lidar_max_range', 12.0)
        self.declare_parameter('lidar_timeout', 0.5)
        self.declare_parameter('stop_if_lidar_lost', True)

        self.declare_parameter('sim_cmd_topic', '/mecanum_drive_controller/cmd_vel')
        self.declare_parameter('sim_use_twist_stamped', True)
        self.declare_parameter('debug_period', 1.0)

        # Slew limiter để tránh lật khi đổi lệnh quá nhanh.
        self.declare_parameter('use_slew_limiter', True)
        self.declare_parameter('max_ax', 0.45)   # m/s^2
        self.declare_parameter('max_ay', 0.45)   # m/s^2
        self.declare_parameter('max_aw', 1.00)   # rad/s^2

        self.topic = str(self.get_parameter('topic').value)
        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.max_wz = float(self.get_parameter('max_wz').value)
        self.pwm_div = float(self.get_parameter('pwm_div').value)
        self.rate = float(self.get_parameter('rate').value)
        self.timeout = float(self.get_parameter('timeout').value)

        self.gains = np.array([
            float(self.get_parameter('gain1').value),
            float(self.get_parameter('gain2').value),
            float(self.get_parameter('gain3').value),
            float(self.get_parameter('gain4').value),
        ], dtype=float)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.lidar_model = str(self.get_parameter('lidar_model').value)
        self.lidar_frame = str(self.get_parameter('lidar_frame').value)
        self.lidar_min_range = float(self.get_parameter('lidar_min_range').value)
        self.lidar_max_range = float(self.get_parameter('lidar_max_range').value)
        self.lidar_timeout = float(self.get_parameter('lidar_timeout').value)
        self.stop_if_lidar_lost = bool(self.get_parameter('stop_if_lidar_lost').value)

        self.sim_cmd_topic = str(self.get_parameter('sim_cmd_topic').value)
        self.sim_use_twist_stamped = bool(self.get_parameter('sim_use_twist_stamped').value)
        self.debug_period = float(self.get_parameter('debug_period').value)

        self.use_slew_limiter = bool(self.get_parameter('use_slew_limiter').value)
        self.max_ax = float(self.get_parameter('max_ax').value)
        self.max_ay = float(self.get_parameter('max_ay').value)
        self.max_aw = float(self.get_parameter('max_aw').value)

        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_wz = 0.0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.last_time = self.get_clock().now()
        self.last_loop_time = self.get_clock().now()
        self.last_scan_time = None
        self.last_scan = None
        self.last_phi = np.zeros(4, dtype=float)
        self.last_debug_time = self.get_clock().now()

        self.sub = self.create_subscription(Twist, self.topic, self.cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)

        if self.sim_use_twist_stamped:
            self.sim_pub_stamped = self.create_publisher(TwistStamped, self.sim_cmd_topic, 10)
            self.sim_pub_twist = None
        else:
            self.sim_pub_twist = self.create_publisher(Twist, self.sim_cmd_topic, 10)
            self.sim_pub_stamped = None

        self.get_logger().info(
            f'Sub {self.topic} -> {self.sim_cmd_topic} | '
            f'max_vx={self.max_vx}, max_vy={self.max_vy}, max_wz={self.max_wz} | '
            f'slew={self.use_slew_limiter}'
        )

    def stop_all(self):
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_wz = 0.0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.publish_sim_command(0.0, 0.0, 0.0)
        self.last_phi[:] = 0.0

    def control(self, phi1, phi2, phi3, phi4):
        self.last_phi[:] = [clamp(phi1), clamp(phi2), clamp(phi3), clamp(phi4)]

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def sector_min(self, scan: LaserScan, start_deg: float, end_deg: float) -> float:
        vals = []
        start = math.radians(start_deg)
        end = math.radians(end_deg)
        n = len(scan.ranges)
        for i in range(n):
            ang = scan.angle_min + i * scan.angle_increment
            if start <= ang <= end:
                r = scan.ranges[i]
                if math.isfinite(r) and self.lidar_min_range <= r <= self.lidar_max_range:
                    vals.append(r)
        if not vals:
            return float('inf')
        return min(vals)

    def cb(self, msg: Twist):
        x_n = clamp(msg.linear.x)
        y_n = clamp(msg.linear.y)
        z_n = clamp(msg.angular.z)
        self.target_vx = x_n * self.max_vx
        self.target_vy = y_n * self.max_vy
        self.target_wz = z_n * self.max_wz
        self.last_time = self.get_clock().now()

    def publish_sim_command(self, vx: float, vy: float, wz: float):
        if self.sim_use_twist_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x = vx
            msg.twist.linear.y = vy
            msg.twist.angular.z = wz
            self.sim_pub_stamped.publish(msg)
        else:
            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = vy
            msg.angular.z = wz
            self.sim_pub_twist.publish(msg)

    def maybe_log(self, text: str):
        now = self.get_clock().now()
        if (now - self.last_debug_time).nanoseconds * 1e-9 >= self.debug_period:
            self.get_logger().info(text)
            self.last_debug_time = now

    def on_timer(self):
        age = (self.get_clock().now() - self.last_time).nanoseconds * 1e-9
        if age > self.timeout:
            self.stop_all()
            self.maybe_log('Timeout /cmd_vel -> STOP')
            return

        if self.stop_if_lidar_lost:
            if self.last_scan_time is None:
                self.stop_all()
                self.maybe_log('Chưa có /scan -> STOP')
                return
            scan_age = (self.get_clock().now() - self.last_scan_time).nanoseconds * 1e-9
            if scan_age > self.lidar_timeout:
                self.stop_all()
                self.maybe_log('Mất /scan -> STOP')
                return

        loop_now = self.get_clock().now()
        dt = max(1e-3, (loop_now - self.last_loop_time).nanoseconds * 1e-9)
        self.last_loop_time = loop_now

        if self.use_slew_limiter:
            self.cmd_vx = slew(self.cmd_vx, self.target_vx, self.max_ax * dt)
            self.cmd_vy = slew(self.cmd_vy, self.target_vy, self.max_ay * dt)
            self.cmd_wz = slew(self.cmd_wz, self.target_wz, self.max_aw * dt)
        else:
            self.cmd_vx = self.target_vx
            self.cmd_vy = self.target_vy
            self.cmd_wz = self.target_wz

        Vg = np.array([[self.cmd_vx], [self.cmd_vy], [self.cmd_wz]], dtype=float)
        phi = (self.J2_inv @ self.J1 @ Vg) / self.pwm_div
        phi = phi.flatten() * self.gains
        phi = np.clip(phi, -1.0, 1.0)
        self.control(phi[0], phi[1], phi[2], phi[3])

        self.publish_sim_command(self.cmd_vx, self.cmd_vy, self.cmd_wz)

        front = None
        if self.last_scan is not None:
            front = self.sector_min(self.last_scan, -15.0, 15.0)
        front_txt = 'inf' if front is None or not math.isfinite(front) else f'{front:.2f}'

        self.maybe_log(
            f'target=({self.target_vx:.2f},{self.target_vy:.2f},{self.target_wz:.2f}) '
            f'cmd=({self.cmd_vx:.2f},{self.cmd_vy:.2f},{self.cmd_wz:.2f}) '
            f'phi=[{self.last_phi[0]:.2f},{self.last_phi[1]:.2f},{self.last_phi[2]:.2f},{self.last_phi[3]:.2f}] '
            f'front={front_txt}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = OmniRobotSimSub()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
