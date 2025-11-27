#!/usr/bin/env python3
import os
import time
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import PoseWithCovarianceStamped

def euler_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class F1TenthEnv(gym.Env):
    def __init__(self, node):
        super().__init__()
        self.node = node
        
        # 차량 설정
        self.max_steer = 0.4
        self.min_speed = 1.0
        self.max_speed = 5.0
        self.max_scan_range = 10.0
        self.n_beams = 20
        self.lane_half_width = 2.0 # 트랙 폭 여유 (너무 좁으면 안됨)

        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_steer, self.min_speed], dtype=np.float32),
            high=np.array([self.max_steer, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )
        
        # 관측: 라이다(20) + 속도(1) + 횡방향오차(1) + 헤딩오차(1) + 미래곡률(1)
        self.obs_dim = self.n_beams + 4
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.prev_steer = 0.0
        self.prev_waypoint_idx = 0
        self.total_waypoints = 0
        self.step_count = 0
        self.steer_smooth_alpha = 0.6

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # [수정 1] Path 데이터가 들어올 때까지 대기 (Total Waypoints 0 방지)
        while self.node.path is None:
            self.node.get_logger().info("Waiting for center path...", throttle_duration_sec=2)
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
        # Path가 확보되었으므로 전체 길이 설정
        self.total_waypoints = len(self.node.path.poses)

        # 정지 명령
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        stop_msg.header.stamp = self.node.get_clock().now().to_msg()
        self.node.pub_drive.publish(stop_msg)

        # 위치 초기화 (ICRA 2025)
        init_pose = PoseWithCovarianceStamped()
        init_pose.header.frame_id = "map"
        init_pose.header.stamp = self.node.get_clock().now().to_msg()
        init_pose.pose.pose.position.x = -14.2865
        init_pose.pose.pose.position.y = -9.1888
        init_pose.pose.pose.position.z = 0.0
        yaw = 1.5623
        init_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
        init_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        for _ in range(6):
            self.node.pub_init_pose.publish(init_pose)
            time.sleep(0.02)
            
        self.node.collision = False
        self.prev_steer = 0.0
        
        # 웨이포인트 인덱스 초기화
        _, _, _, current_idx = self._get_track_data()
        self.prev_waypoint_idx = current_idx
        
        # [수정 3] step_count는 리셋 마지막에 0으로
        self.step_count = 0
        
        time.sleep(0.2)
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
        return self._get_obs(), {}

    def step(self, action):
        # Action 처리
        raw_steer = float(np.clip(action[0], -self.max_steer, self.max_steer))
        raw_speed = float(np.clip(action[1], self.min_speed, self.max_speed))
        
        steer = self.steer_smooth_alpha * self.prev_steer + (1.0 - self.steer_smooth_alpha) * raw_steer
        
        # 자동 감속
        _, _, curv, _ = self._get_track_data()
        k_curv = 2.5
        desired_speed = max(self.min_speed, self.max_speed * math.exp(-k_curv * abs(curv)))
        speed = min(raw_speed, desired_speed)
        
        msg = AckermannDriveStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        msg.drive.acceleration = 2.0
        self.node.pub_drive.publish(msg)

        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self._get_obs()
        lat_err, head_err, _, current_idx = self._get_track_data()
        reward, terminated = self._get_reward(obs, (steer, speed), lat_err, head_err, current_idx)
        
        if terminated:
            stop = AckermannDriveStamped()
            stop.header.stamp = self.node.get_clock().now().to_msg()
            stop.drive.speed = 0.0
            self.node.pub_drive.publish(stop)
            
        self.prev_steer = steer
        self.prev_waypoint_idx = current_idx
        self.step_count += 1
        
        return obs, float(reward), bool(terminated), False, {}

    def _get_obs(self):
        speed = self.node.odom.twist.twist.linear.x if self.node.odom else 0.0
        speed_norm = np.clip(speed / self.max_speed, 0.0, 1.0)
        
        scan_data = np.ones(self.n_beams) * self.max_scan_range
        if self.node.scan and len(self.node.scan.ranges) > 0:
            raw_ranges = np.array(self.node.scan.ranges)
            raw_ranges = np.nan_to_num(raw_ranges, posinf=self.max_scan_range, neginf=0.0)
            stride = max(1, len(raw_ranges) // self.n_beams)
            for i in range(self.n_beams):
                start = i * stride
                end = min((i + 1) * stride, len(raw_ranges))
                if start < end:
                    scan_data[i] = np.min(raw_ranges[start:end])
        scan_norm = np.clip(scan_data / self.max_scan_range, 0.0, 1.0)
        
        lat_err, head_err, curv, _ = self._get_track_data()
        
        # 정규화 (범위를 넉넉하게 잡음)
        lat_norm = np.clip(lat_err / 3.0, -1.0, 1.0) 
        head_norm = np.clip(head_err / math.pi, -1.0, 1.0)
        curv_norm = np.clip(curv / math.pi, -1.0, 1.0)
        
        obs = np.concatenate(([speed_norm], scan_norm, [lat_norm, head_norm, curv_norm])).astype(np.float32)
        return obs

    def _get_reward(self, obs, action, lat_err, head_err, current_idx):
        speed_norm = float(obs[0])
        scan_data = obs[1:1 + self.n_beams]
        steer, speed = action
        
        # 1. 충돌 (패널티 완화)
        if self.node.collision:
            # 초반 스텝 보호 로직 삭제 (어차피 학습되면 알아서 피함)
            self.node.get_logger().info("CRASH: -10.0")
            return -10.0, True
            
        # 2. 이탈 (패널티 완화)
        if abs(lat_err) > 2.5: # 2.5m까지 허용
            return -10.0, True

        reward = 0.0
        
        # A. 진행 보상 (핵심)
        diff = current_idx - self.prev_waypoint_idx
        
        # [수정 2] 완주 감지 (Loop Closure)
        # 인덱스가 갑자기 크게 줄어들면 (예: 990 -> 10) 한 바퀴 돈 것임
        if diff < - (self.total_waypoints // 2):
            diff += self.total_waypoints
            self.node.get_logger().info(f"LAP COMPLETED! Reward Bonus! (Time: {self.node.current_lap_time:.2f})")
            reward += 200.0 # 완주 축하금
            
        elif diff > (self.total_waypoints // 2): # 뒤로 돈 경우
            diff -= self.total_waypoints
            
        # 앞으로 가면 점수 (기존 2.5 -> 3.0 강화)
        if diff > 0:
            reward += float(diff) * 3.0 
            
        # B. 속도 보상
        reward += speed_norm * 0.5
        
        # C. 페널티들 (약하게)
        reward -= 0.01 # 시간
        
        steer_diff = abs(steer - self.prev_steer)
        reward -= steer_diff * 2.0 # 흔들림 방지
        
        vy = self.node.odom.twist.twist.linear.y if self.node.odom else 0.0
        reward -= abs(vy) * 2.0 # 미끄러짐 방지

        # D. 벽 근처 위험
        min_dist = float(np.min(scan_data)) * self.max_scan_range
        if min_dist < 0.5:
            reward -= (0.5 - min_dist) * 5.0

        return float(reward), False

    def _get_track_data(self):
        if not self.node.path or not self.node.odom:
            return 0.0, 0.0, 0.0, 0
            
        car_x = self.node.odom.pose.pose.position.x
        car_y = self.node.odom.pose.pose.position.y
        car_yaw = euler_from_quaternion(self.node.odom.pose.pose.orientation)
        poses = self.node.path.poses
        n = len(poses)
        
        # Window Search
        window_size = 100
        start = max(0, self.prev_waypoint_idx - window_size//2)
        end = min(n, self.prev_waypoint_idx + window_size//2)
        
        min_dist_sq = float('inf')
        closest_idx = self.prev_waypoint_idx
        
        for i in range(start, end):
            p = poses[i].pose.position
            d2 = (p.x - car_x)**2 + (p.y - car_y)**2
            if d2 < min_dist_sq:
                min_dist_sq = d2
                closest_idx = i
                
        # Global Search (Fallback)
        if min_dist_sq > 25.0:
            min_dist_sq = float('inf')
            for i, pwr in enumerate(poses):
                p = pwr.pose.position
                d2 = (p.x - car_x)**2 + (p.y - car_y)**2
                if d2 < min_dist_sq:
                    min_dist_sq = d2
                    closest_idx = i
                    
        def find_ahead_idx(start_idx, lookahead_m):
            accum = 0.0
            idx = start_idx
            for _ in range(n):
                a = poses[idx % n].pose.position
                b = poses[(idx + 1) % n].pose.position
                accum += math.hypot(b.x - a.x, b.y - a.y)
                idx += 1
                if accum >= lookahead_m:
                    return idx % n
            return start_idx

        next_idx = find_ahead_idx(closest_idx, 1.5)
        far_idx = find_ahead_idx(closest_idx, 5.0)
        
        a = poses[closest_idx].pose.position
        b = poses[next_idx].pose.position
        
        track_yaw = math.atan2(b.y - a.y, b.x - a.x)
        heading_error = track_yaw - car_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        
        dx_vec = car_x - a.x
        dy_vec = car_y - a.y
        lateral_error = -math.sin(track_yaw) * dx_vec + math.cos(track_yaw) * dy_vec
        
        fa = poses[far_idx].pose.position
        curvature = math.atan2(fa.y - b.y, fa.x - b.x) - track_yaw
        curvature = (curvature + math.pi) % (2 * math.pi) - math.pi
        
        return lateral_error, heading_error, curvature, closest_idx

class RLNode(Node):
    def __init__(self):
        super().__init__('rl_driver_node')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        
        self.pub_drive = self.create_publisher(AckermannDriveStamped, 'ackermann_cmd0', qos)
        self.pub_init_pose = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', qos_reliable)
        
        self.create_subscription(Odometry, 'odom0', self.odom_cb, qos)
        self.create_subscription(LaserScan, 'scan0', self.scan_cb, qos)
        self.create_subscription(Bool, 'collision0', self.collision_cb, qos)
        self.create_subscription(Path, 'center_path', self.path_cb, qos_reliable)
        self.create_subscription(Float64, 'current_lap_time', self.lap_time_cb, qos)
        
        self.odom = None
        self.scan = None
        self.collision = False
        self.path = None
        self.current_lap_time = 0.0
        
        self.get_logger().info("Waiting for path & odom...")
        while rclpy.ok() and (self.path is None or self.odom is None):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Ready!")

    def odom_cb(self, msg): self.odom = msg
    def scan_cb(self, msg): self.scan = msg
    def collision_cb(self, msg): self.collision = bool(msg.data)
    def path_cb(self, msg): self.path = msg
    def lap_time_cb(self, msg): self.current_lap_time = msg.data

def main():
    rclpy.init()
    node = RLNode()
    
    def make_env():
        env = F1TenthEnv(node)
        return Monitor(env)
        
    vecenv = DummyVecEnv([make_env])
    vecenv = VecFrameStack(vecenv, n_stack=4)
    
    if not os.path.exists("./models/"):
        os.makedirs("./models/")
    
    # [수정 4] Batch size 최적화 (CPU 학습용)
    # 512는 너무 큽니다. 128 정도로 줄여서 업데이트 빈도를 높이는 게 좋습니다.
    model = PPO("MlpPolicy", vecenv, verbose=1,
                learning_rate=5e-5,
                ent_coef=0.01,
                batch_size=128,  # 512 -> 128
                n_steps=2048,
                n_epochs=10,
                clip_range=0.2,  # 0.1 -> 0.2 (표준)
                gae_lambda=0.95,
                gamma=0.995,
                device='cpu')
                
    checkpoint = CheckpointCallback(save_freq=20000, save_path='./models/', name_prefix='f1tenth')
    
    print("START: Fixed logic + Lap Bonus")
    
    try:
        model.learn(total_timesteps=2000000, callback=checkpoint)
    except KeyboardInterrupt:
        model.save("f1tenth_interrupted")
    finally:
        model.save("f1tenth_final")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
