#!/usr/bin/env python3
"""
Publisher phía PC: đọc /scan và phát /cmd_vel chuẩn hoá cho subscriber phía xe.

Bản này là RIGHT-HAND WALL FOLLOWING + MEMORY:
- Vẫn bám tường phải bằng LiDAR.
- Bổ sung "ghi nhớ giao lộ / ô đã đi qua" để giảm việc quay ngược về ngõ vào.
- Không dùng cảm biến nào ngoài LiDAR; phần nhớ vị trí là ước lượng nội bộ từ chính
  lệnh vận tốc đã gửi đi (dead-reckoning mở vòng), KHÔNG dùng odom.

Ý tưởng bộ nhớ:
- Lượng tử hoá không gian thành các ô vuông `memory_cell_size`.
- Ghi lại:
    * số lần đã ghé mỗi ô
    * số lần đã đi qua mỗi cạnh giữa hai ô
- Khi tới chỗ có lựa chọn (trái / thẳng / phải), xe chấm điểm các nhánh mở:
    score = bonus_nhánh_mới + bonus_ưu_tiên_bám_phải + thưởng_khoảng_trống
            - phạt_ô_đã_thăm - phạt_cạnh_đã_đi
- Vì cạnh từ ngõ vào -> trong mê cung đã được đánh dấu, khi quay lại cửa vào,
  nhánh đi ngược ra ngoài sẽ bị phạt mạnh hơn và chỉ chọn khi thật sự bế tắc.

Bản vá ENTRY GUARD:
- Không đổi logic cơ bản bám phải + memory.
- Chỉ bổ sung một lớp "chống lao ra ngoài ở ngõ vào":
    * trước khi xác nhận đã vào trong mê cung, phạt rất mạnh các nhánh quá thoáng
      kiểu khoảng sân bên ngoài
    * ưu tiên đi thẳng xuyên qua cửa mê cung nếu phía trước còn an toàn
    * khi đang ở ngoài và chưa thấy tường phải, KHÔNG quay phải tìm tường như cũ
      mà ưu tiên tiến thẳng / hơi ép trái để chui vào cửa
    * sau khi đã xác nhận vào trong mê cung, tiếp tục phạt các ô/cạnh thuộc vùng ngoài
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


Cell = Tuple[int, int]


def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class MazeLidarPublisher(Node):
    def __init__(self):
        super().__init__('maze_lidar_publisher_right_memory')

        # ===== Giao tiếp ROS =====
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('rate', 20.0)

        # ===== Giới hạn động học - đồng bộ subscriber/xe =====
        self.declare_parameter('max_vx', 0.5)
        self.declare_parameter('max_vy', 0.5)
        self.declare_parameter('max_wz', 1.5)

        # ===== LiDAR =====
        self.declare_parameter('lidar_min_range', 0.05)
        self.declare_parameter('lidar_max_range', 12.0)
        self.declare_parameter('scan_timeout', 0.5)

        # ===== Bám phải + điều khiển =====
        self.declare_parameter('desired_wall_distance', 0.22)
        self.declare_parameter('lookahead', 0.12)
        self.declare_parameter('front_block_distance', 0.30)
        self.declare_parameter('front_safe_distance', 0.42)
        self.declare_parameter('left_open_distance', 0.45)
        self.declare_parameter('right_open_distance', 0.45)
        self.declare_parameter('linear_speed', 0.14)
        self.declare_parameter('slow_linear_speed', 0.07)
        self.declare_parameter('turn_speed', 0.72)
        self.declare_parameter('kp', 3.0)
        self.declare_parameter('kd', 0.45)
        self.declare_parameter('turn_left_hold', 0.34)
        self.declare_parameter('turn_right_hold', 0.28)
        self.declare_parameter('u_turn_hold', 0.62)
        self.declare_parameter('decision_cooldown', 0.75)
        self.declare_parameter('debug_period', 0.5)

        # ===== Bộ nhớ đường đi =====
        self.declare_parameter('memory_cell_size', 0.45)
        self.declare_parameter('memory_new_cell_bonus', 2.5)
        self.declare_parameter('memory_new_edge_bonus', 1.8)
        self.declare_parameter('memory_visit_penalty', 0.9)
        self.declare_parameter('memory_edge_penalty', 1.6)
        self.declare_parameter('left_preference_bonus', 0.05)
        self.declare_parameter('front_preference_bonus', 0.20)
        self.declare_parameter('right_preference_bonus', 0.35)
        self.declare_parameter('back_penalty', 4.0)
        self.declare_parameter('distance_reward_gain', 0.20)
        self.declare_parameter('junction_detect_distance', 0.50)
        self.declare_parameter('memory_enable', True)

        # ===== Entry / exit guard: tránh đi ngược ra ngoài ở cửa mê cung =====
        self.declare_parameter('entry_guard_enable', True)
        self.declare_parameter('outside_open_distance', 1.60)
        self.declare_parameter('outside_branch_penalty', 3.20)
        self.declare_parameter('outside_cell_penalty', 2.40)
        self.declare_parameter('entry_forward_bonus', 1.10)
        self.declare_parameter('entry_confirm_cycles', 8)
        self.declare_parameter('corridor_side_max', 0.85)
        self.declare_parameter('corridor_front_max', 1.20)
        self.declare_parameter('entry_bias_left_wz', 0.10)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)
        self.rate = float(self.get_parameter('rate').value)

        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.max_wz = float(self.get_parameter('max_wz').value)

        self.lidar_min_range = float(self.get_parameter('lidar_min_range').value)
        self.lidar_max_range = float(self.get_parameter('lidar_max_range').value)
        self.scan_timeout = float(self.get_parameter('scan_timeout').value)

        self.desired_wall_distance = float(self.get_parameter('desired_wall_distance').value)
        self.lookahead = float(self.get_parameter('lookahead').value)
        self.front_block_distance = float(self.get_parameter('front_block_distance').value)
        self.front_safe_distance = float(self.get_parameter('front_safe_distance').value)
        self.left_open_distance = float(self.get_parameter('left_open_distance').value)
        self.right_open_distance = float(self.get_parameter('right_open_distance').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.slow_linear_speed = float(self.get_parameter('slow_linear_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.turn_left_hold = float(self.get_parameter('turn_left_hold').value)
        self.turn_right_hold = float(self.get_parameter('turn_right_hold').value)
        self.u_turn_hold = float(self.get_parameter('u_turn_hold').value)
        self.decision_cooldown = float(self.get_parameter('decision_cooldown').value)
        self.debug_period = float(self.get_parameter('debug_period').value)

        self.memory_cell_size = float(self.get_parameter('memory_cell_size').value)
        self.memory_new_cell_bonus = float(self.get_parameter('memory_new_cell_bonus').value)
        self.memory_new_edge_bonus = float(self.get_parameter('memory_new_edge_bonus').value)
        self.memory_visit_penalty = float(self.get_parameter('memory_visit_penalty').value)
        self.memory_edge_penalty = float(self.get_parameter('memory_edge_penalty').value)
        self.left_preference_bonus = float(self.get_parameter('left_preference_bonus').value)
        self.front_preference_bonus = float(self.get_parameter('front_preference_bonus').value)
        self.right_preference_bonus = float(self.get_parameter('right_preference_bonus').value)
        self.back_penalty = float(self.get_parameter('back_penalty').value)
        self.distance_reward_gain = float(self.get_parameter('distance_reward_gain').value)
        self.junction_detect_distance = float(self.get_parameter('junction_detect_distance').value)
        self.memory_enable = bool(self.get_parameter('memory_enable').value)

        self.entry_guard_enable = bool(self.get_parameter('entry_guard_enable').value)
        self.outside_open_distance = float(self.get_parameter('outside_open_distance').value)
        self.outside_branch_penalty = float(self.get_parameter('outside_branch_penalty').value)
        self.outside_cell_penalty = float(self.get_parameter('outside_cell_penalty').value)
        self.entry_forward_bonus = float(self.get_parameter('entry_forward_bonus').value)
        self.entry_confirm_cycles = int(self.get_parameter('entry_confirm_cycles').value)
        self.corridor_side_max = float(self.get_parameter('corridor_side_max').value)
        self.corridor_front_max = float(self.get_parameter('corridor_front_max').value)
        self.entry_bias_left_wz = float(self.get_parameter('entry_bias_left_wz').value)

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)

        self.last_scan: Optional[LaserScan] = None
        self.last_scan_time = None

        self.prev_error = 0.0
        self.prev_control_time = None

        self.mode = 'IDLE'
        self.mode_until = None
        self.mode_cmd = (0.0, 0.0, 0.0)

        self.last_debug_time = self.get_clock().now()
        self.last_decision_time = -1e9

        # ===== Ước lượng pose nội bộ từ lệnh đã gửi =====
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.last_motion_update_time = self.get_clock().now().nanoseconds * 1e-9
        self.sent_vx = 0.0
        self.sent_vy = 0.0
        self.sent_wz = 0.0
        self.current_cell: Cell = self.quantize_cell(self.pose_x, self.pose_y)
        self.visited_cells: Counter = Counter({self.current_cell: 1})
        self.visited_edges: Counter = Counter()
        self.entered_maze = False
        self.corridor_seen_count = 0
        self.outside_cells = {self.current_cell}

        self.get_logger().info(
            'Maze solver RIGHT+MEMORY chạy | '
            f'scan={self.scan_topic} -> cmd={self.cmd_topic} | '
            'bám phải + geometric PD + nhớ ô/cạnh đã đi + entry_guard'
        )

    # ------------------------- LiDAR helpers -------------------------
    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def _wrap_to_pi(self, a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _valid_values(self, values: List[float]) -> List[float]:
        out = []
        for v in values:
            if math.isfinite(v) and self.lidar_min_range <= v <= self.lidar_max_range:
                out.append(v)
        return out

    def range_at(self, scan: LaserScan, angle_deg: float, window_deg: float = 8.0) -> float:
        center = math.radians(angle_deg)
        half = math.radians(window_deg / 2.0)
        values = []
        n = len(scan.ranges)
        for i in range(n):
            ang = scan.angle_min + i * scan.angle_increment
            if abs(self._wrap_to_pi(ang - center)) <= half:
                values.append(scan.ranges[i])
        vals = self._valid_values(values)
        if not vals:
            return float('inf')
        vals.sort()
        return vals[len(vals) // 2]

    def sector_min(self, scan: LaserScan, start_deg: float, end_deg: float) -> float:
        start = math.radians(start_deg)
        end = math.radians(end_deg)
        values = []
        n = len(scan.ranges)
        for i in range(n):
            ang = scan.angle_min + i * scan.angle_increment
            if start <= ang <= end:
                values.append(scan.ranges[i])
        vals = self._valid_values(values)
        if not vals:
            return float('inf')
        return min(vals)

    # ----------------------- Command / motion ------------------------
    def publish_norm_cmd(self, x_norm: float, y_norm: float, z_norm: float):
        msg = Twist()
        msg.linear.x = clamp(x_norm)
        msg.linear.y = clamp(y_norm)
        msg.angular.z = clamp(z_norm)
        self.pub.publish(msg)

    def publish_physical_cmd(self, vx: float, vy: float, wz: float):
        self.sent_vx = vx
        self.sent_vy = vy
        self.sent_wz = wz
        x_norm = vx / self.max_vx if self.max_vx > 1e-9 else 0.0
        y_norm = vy / self.max_vy if self.max_vy > 1e-9 else 0.0
        z_norm = wz / self.max_wz if self.max_wz > 1e-9 else 0.0
        self.publish_norm_cmd(x_norm, y_norm, z_norm)

    def set_mode(self, mode: str, hold_s: float, vx: float, vy: float, wz: float):
        self.mode = mode
        self.mode_until = self.get_clock().now().nanoseconds * 1e-9 + hold_s
        self.mode_cmd = (vx, vy, wz)

    def maybe_log(self, text: str):
        now = self.get_clock().now()
        if (now - self.last_debug_time).nanoseconds * 1e-9 >= self.debug_period:
            self.get_logger().info(text)
            self.last_debug_time = now

    def integrate_motion_memory(self):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        dt = max(0.0, now_s - self.last_motion_update_time)
        self.last_motion_update_time = now_s
        if dt <= 0.0:
            return

        c = math.cos(self.pose_yaw)
        s = math.sin(self.pose_yaw)
        vx_w = self.sent_vx * c - self.sent_vy * s
        vy_w = self.sent_vx * s + self.sent_vy * c
        prev_cell = self.current_cell

        self.pose_x += vx_w * dt
        self.pose_y += vy_w * dt
        self.pose_yaw = self._wrap_to_pi(self.pose_yaw + self.sent_wz * dt)
        self.current_cell = self.quantize_cell(self.pose_x, self.pose_y)

        if self.current_cell != prev_cell:
            self.visited_cells[self.current_cell] += 1
            e = self.edge_key(prev_cell, self.current_cell)
            self.visited_edges[e] += 1
        if not self.entered_maze:
            self.outside_cells.add(self.current_cell)

    # -------------------------- Memory -------------------------------
    def quantize_cell(self, x: float, y: float) -> Cell:
        qx = int(round(x / max(self.memory_cell_size, 1e-6)))
        qy = int(round(y / max(self.memory_cell_size, 1e-6)))
        return (qx, qy)

    def heading_index(self) -> int:
        # 0: +X, 1: +Y, 2: -X, 3: -Y
        yaw = self._wrap_to_pi(self.pose_yaw)
        idx = int(round(yaw / (math.pi / 2.0))) % 4
        return idx

    def edge_key(self, a: Cell, b: Cell):
        return tuple(sorted((a, b)))

    def neighbor_for_relative(self, rel: str) -> Cell:
        heading = self.heading_index()
        rel_to_heading = {
            'front': 0,
            'left': 1,
            'back': 2,
            'right': 3,
        }
        abs_heading = (heading + rel_to_heading[rel]) % 4
        dirs = {
            0: (1, 0),   # +X
            1: (0, 1),   # +Y
            2: (-1, 0),  # -X
            3: (0, -1),  # -Y
        }
        dx, dy = dirs[abs_heading]
        return (self.current_cell[0] + dx, self.current_cell[1] + dy)

    def update_entry_status(self, front: float, left: float, right: float, front_right: float):
        """Xác nhận đã thực sự chui vào trong mê cung/corridor."""
        if self.entered_maze:
            return

        side_hits = 0
        for v in (left, right):
            if math.isfinite(v) and v < self.corridor_side_max:
                side_hits += 1

        corridor_like = False
        if side_hits >= 2:
            corridor_like = True
        elif side_hits >= 1 and math.isfinite(front) and front < self.corridor_front_max:
            corridor_like = True
        elif math.isfinite(right) and math.isfinite(front_right):
            if right < self.corridor_side_max and front_right < self.corridor_front_max:
                corridor_like = True

        if corridor_like:
            self.corridor_seen_count += 1
        else:
            self.corridor_seen_count = max(0, self.corridor_seen_count - 1)

        if self.corridor_seen_count >= self.entry_confirm_cycles:
            self.entered_maze = True
            self.maybe_log(
                f'ENTRY_GUARD -> entered_maze=True | current_cell={self.current_cell} outside_cells={len(self.outside_cells)}'
            )

    def choose_branch_with_memory(self, front: float, left: float, right: float) -> str:
        # Candidate branches currently open.
        candidates: Dict[str, float] = {}
        if math.isfinite(left) and left > self.left_open_distance:
            candidates['left'] = left
        if math.isfinite(front) and front > self.front_block_distance:
            candidates['front'] = front
        if math.isfinite(right) and right > self.right_open_distance:
            candidates['right'] = right

        if not candidates:
            return 'back'

        pref_bonus = {
            'left': self.left_preference_bonus,
            'front': self.front_preference_bonus,
            'right': self.right_preference_bonus,
            'back': -self.back_penalty,
        }

        best_rel = 'front'
        best_score = -1e9
        lines = []

        for rel, openness in candidates.items():
            target = self.neighbor_for_relative(rel)
            visits = self.visited_cells[target]
            edge_visits = self.visited_edges[self.edge_key(self.current_cell, target)]
            score = 0.0
            score += pref_bonus.get(rel, 0.0)
            score += self.distance_reward_gain * min(openness, 2.5)
            score += self.memory_new_cell_bonus if visits == 0 else -self.memory_visit_penalty * visits
            score += self.memory_new_edge_bonus if edge_visits == 0 else -self.memory_edge_penalty * edge_visits

            if self.entry_guard_enable:
                if not self.entered_maze:
                    if rel == 'front':
                        score += self.entry_forward_bonus
                    if rel in ('left', 'right') and openness > self.outside_open_distance:
                        score -= self.outside_branch_penalty * min(openness / max(self.outside_open_distance, 1e-6), 2.0)
                    if target in self.outside_cells:
                        score -= self.outside_cell_penalty
                else:
                    if target in self.outside_cells:
                        score -= (self.outside_cell_penalty + 0.75 * self.outside_branch_penalty)
                    if rel in ('left', 'right', 'front') and openness > (self.outside_open_distance + 0.30):
                        score -= 0.50 * self.outside_branch_penalty

            lines.append(f'{rel}:s={score:.2f},cell={target},v={visits},e={edge_visits},open={openness:.2f},in={int(self.entered_maze)}')
            if score > best_score:
                best_score = score
                best_rel = rel

        self.maybe_log('memory {' + ' | '.join(lines) + f'}} -> {best_rel}')
        return best_rel

    # ----------------------- Wall-follow PD --------------------------
    def wall_follow_control(self, a: float, b: float, dt: float):
        theta = math.radians(45.0)
        alpha = math.atan2((a * math.cos(theta) - b), (a * math.sin(theta)))
        d_t = b * math.cos(alpha)
        d_tp1 = d_t + self.lookahead * math.sin(alpha)
        error = self.desired_wall_distance - d_tp1
        d_error = 0.0 if dt <= 1e-6 else (error - self.prev_error) / dt
        self.prev_error = error

        # Bám phải => error dương nghĩa là đang quá gần tường phải, cần wz dương để tách ra.
        wz = (self.kp * error + self.kd * d_error)
        wz = clamp(wz / self.max_wz) * self.max_wz
        vx = self.linear_speed
        return vx, 0.0, wz, error, alpha

    # ----------------------------- Main ------------------------------
    def on_timer(self):
        self.integrate_motion_memory()

        if self.last_scan is None or self.last_scan_time is None:
            self.publish_physical_cmd(0.0, 0.0, 0.0)
            self.maybe_log('Chưa có /scan -> stop')
            return

        age = (self.get_clock().now() - self.last_scan_time).nanoseconds * 1e-9
        if age > self.scan_timeout:
            self.publish_physical_cmd(0.0, 0.0, 0.0)
            self.maybe_log('Mất /scan -> stop')
            return

        scan = self.last_scan
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_control_time is None:
            dt = 1.0 / self.rate
        else:
            dt = max(1e-3, now_s - self.prev_control_time)
        self.prev_control_time = now_s

        front = self.sector_min(scan, -12.0, 12.0)
        left = self.sector_min(scan, 70.0, 110.0)
        front_right = self.range_at(scan, -45.0, 10.0)
        right = self.sector_min(scan, -110.0, -70.0)

        self.update_entry_status(front, left, right, front_right)

        # Giữ mode tạm thời nếu đang quay/rẽ.
        if self.mode_until is not None and now_s < self.mode_until:
            self.publish_physical_cmd(*self.mode_cmd)
            self.maybe_log(
                f'mode={self.mode} hold | cell={self.current_cell} yaw={math.degrees(self.pose_yaw):.0f} '
                f'front={front:.2f} left={left:.2f} right={right:.2f}'
            )
            return
        self.mode = 'IDLE'
        self.mode_until = None

        # ===== Quyết định tại giao lộ bằng memory =====
        left_open = math.isfinite(left) and left > max(self.left_open_distance, self.junction_detect_distance)
        right_open = math.isfinite(right) and right > max(self.right_open_distance, self.junction_detect_distance)
        front_open = math.isfinite(front) and front > self.front_safe_distance
        at_decision = left_open or right_open or (front < self.front_block_distance) or (front_open and right_open)
        can_decide = (now_s - self.last_decision_time) > self.decision_cooldown

        if at_decision and can_decide:
            choice = self.choose_branch_with_memory(front, left, right) if self.memory_enable else (
                'right' if right_open else ('front' if front_open else ('left' if left_open else 'back'))
            )
            self.last_decision_time = now_s

            if choice == 'right' and right_open:
                self.set_mode('TURN_RIGHT_MEMORY', self.turn_right_hold, 0.0, 0.0, -self.turn_speed)
                self.publish_physical_cmd(*self.mode_cmd)
                self.maybe_log(f'choice=RIGHT | cell={self.current_cell} visited={self.visited_cells[self.current_cell]}')
                return
            if choice == 'left' and left_open:
                self.set_mode('TURN_LEFT_MEMORY', self.turn_left_hold, 0.0, 0.0, +self.turn_speed)
                self.publish_physical_cmd(*self.mode_cmd)
                self.maybe_log(f'choice=LEFT | cell={self.current_cell} visited={self.visited_cells[self.current_cell]}')
                return
            if choice == 'back':
                if not self.entered_maze and self.entry_guard_enable and math.isfinite(front) and front > self.front_safe_distance:
                    # Ở ngay cửa mê cung: tránh quay đầu ra ngoài quá sớm, cứ ưu tiên tiến thẳng thêm.
                    self.maybe_log('ENTRY_GUARD veto BACKTRACK -> keep forward')
                else:
                    self.set_mode('U_TURN_MEMORY', self.u_turn_hold, 0.0, 0.0, -self.turn_speed)
                    self.publish_physical_cmd(*self.mode_cmd)
                    self.maybe_log(f'choice=BACKTRACK | cell={self.current_cell} visited={self.visited_cells[self.current_cell]}')
                    return
            # choice == front -> không cần set mode, đi tiếp xuống dưới

        # Trước mặt bí hẳn -> quay trái tìm lối (tương thích bám phải).
        if math.isfinite(front) and front < self.front_block_distance:
            self.set_mode('TURN_LEFT_BLOCKED', self.turn_left_hold, 0.0, 0.0, +self.turn_speed)
            self.publish_physical_cmd(*self.mode_cmd)
            self.maybe_log(f'mode=TURN_LEFT_BLOCKED | front={front:.2f}')
            return

        # Chưa nhìn thấy đủ tường phải.
        if (not math.isfinite(right)) or (not math.isfinite(front_right)):
            if self.entry_guard_enable and (not self.entered_maze):
                # Ở ngoài cửa mê cung: không quét phải như cũ vì dễ ôm khoảng trống bên ngoài.
                # Thay vào đó ưu tiên tiến thẳng xuyên qua cửa, hoặc hơi ép trái để tránh drift ra ngoài.
                vx = self.linear_speed if (math.isfinite(front) and front > self.front_safe_distance) else self.slow_linear_speed
                vy = 0.0
                wz = 0.0
                if (not math.isfinite(right)) or right > self.outside_open_distance:
                    wz = +self.entry_bias_left_wz * self.turn_speed
                if math.isfinite(front) and front < self.front_block_distance:
                    wz = +0.45 * self.turn_speed
                    vx = 0.0
                self.publish_physical_cmd(vx, vy, wz)
                self.maybe_log(
                    f'mode=ENTRY_SEEK_CORRIDOR | in={int(self.entered_maze)} front={front:.2f} left={left:.2f} fr={front_right:.2f} right={right:.2f}'
                )
                return
            vx = self.slow_linear_speed
            wz = -0.35 * self.turn_speed
            self.publish_physical_cmd(vx, 0.0, wz)
            self.maybe_log('mode=SEARCH_RIGHT_WALL')
            return

        # Theo tường phải bằng geometric PD.
        vx, vy, wz, error, alpha = self.wall_follow_control(front_right, right, dt)
        if math.isfinite(front) and front < self.front_safe_distance:
            ratio = (front - self.front_block_distance) / max(
                self.front_safe_distance - self.front_block_distance, 1e-6
            )
            ratio = clamp(ratio, 0.25, 1.0)
            vx *= ratio

        self.publish_physical_cmd(vx, vy, wz)
        self.maybe_log(
            'mode=FOLLOW_RIGHT_MEMORY | '
            f'in={int(self.entered_maze)} cell={self.current_cell} pose=({self.pose_x:.2f},{self.pose_y:.2f},{math.degrees(self.pose_yaw):.0f}deg) '
            f'front={front:.2f} fr={front_right:.2f} left={left:.2f} right={right:.2f} '
            f'err={error:.3f} alpha={math.degrees(alpha):.1f}deg vx={vx:.2f} wz={wz:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MazeLidarPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
