#!/usr/bin/env python3
import os
import time
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped

def euler_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class F1TenthEnv(gym.Env):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.max_steer = 0.4
        self.min_speed = 1.0
        self.max_speed = 5.0
        
        # 라이다 빔 개수 (학습 속도 향상용 다운샘플링)
        self.n_beams = 20 

        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_steer, self.min_speed]),
            high=np.array([self.max_steer, self.max_speed]),
            dtype=np.float32
        )
        
        # 관측: 라이다(20) + 속도(1) + 횡방향오차(1) + 헤딩오차(1) + 미래곡률(1) = 24개
        self.obs_dim = self.n_beams + 4
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.prev_steer = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 정지 명령
        stop_msg = AckermannDriveStamped()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        self.node.pub_drive.publish(stop_msg)

        # 2. 위치 초기화 (icra2025 맵 시작 좌표)
        init_pose = PoseWithCovarianceStamped()
        init_pose.header.frame_id = "map"
        init_pose.header.stamp = self.node.get_clock().now().to_msg()
        init_pose.pose.pose.position.x = -14.2865
        init_pose.pose.pose.position.y = -9.1888
        init_pose.pose.pose.position.z = 0.0
        
        # Yaw -> Quaternion 변환
        yaw = 1.5623
        init_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
        init_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.node.pub_init_pose.publish(init_pose)
        
        # 3. 상태 초기화
        self.node.collision = False
        self.prev_steer = 0.0
        self.node.get_logger().info("RESET: 차량 위치 초기화")
        
        time.sleep(0.5) 
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        return self._get_obs(), {}

    def step(self, action):
        steer = float(np.clip(action[0], -self.max_steer, self.max_steer))
        speed = float(np.clip(action[1], self.min_speed, self.max_speed))

        msg = AckermannDriveStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        msg.drive.acceleration = 2.0
        self.node.pub_drive.publish(msg)

        # 데이터 갱신 대기
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.01)

        obs = self._get_obs()
        reward, terminated = self._get_reward(obs, action)
        
        self.prev_steer = steer
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # 1. 속도
        speed = self.node.odom.twist.twist.linear.x if self.node.odom else 0.0
        
        # 2. 라이다 (20개로 축소)
        scan_data = np.ones(self.n_beams) * 10.0
        if self.node.scan and len(self.node.scan.ranges) > 0:
            raw_ranges = np.array(self.node.scan.ranges)
            raw_ranges = np.nan_to_num(raw_ranges, posinf=10.0, neginf=0.0)
            
            stride = max(1, len(raw_ranges) // self.n_beams)
            for i in range(self.n_beams):
                # 인덱스 범위 초과 방지
                start = i * stride
                end = min((i + 1) * stride, len(raw_ranges))
                if start < end:
                    scan_data[i] = np.min(raw_ranges[start:end])

        # 3. 트랙 정보
        lat_err, head_err, lookahead_curv = self._get_track_data()
        
        # 4. 벡터 합치기
        obs = np.concatenate(([speed], scan_data, [lat_err, head_err, lookahead_curv])).astype(np.float32)
        return obs

    def _get_reward(self, obs, action):
        # obs 구조: [speed(1), scan(20), lat(1), head(1), curv(1)]
        speed = obs[0]
        # scan = obs[1:21] # 필요시 사용
        lat_err = obs[-3]
        head_err = obs[-2]
        
        # 1. 충돌
        if self.node.collision:
            print("💥 CRASH! (-200)")
            return -200.0, True

        # 2. 트랙 이탈
        if abs(lat_err) > 1.2: # 조금 여유 있게 1.2m
            print(f"OUT OF TRACK: {lat_err:.2f} (-100)")
            return -100.0, True

        # 3. 보상 계산
        reward = speed * 2.0
        reward -= abs(lat_err) * 4.0
        reward -= abs(head_err) * 2.0
        
        # 조향 변화 페널티 (부드러운 주행)
        steer_diff = abs(action[0] - self.prev_steer)
        reward -= steer_diff * 5.0

        return reward, False

    def _get_track_data(self):
        if not self.node.path or not self.node.odom:
            return 0.0, 0.0, 0.0

        car_x = self.node.odom.pose.pose.position.x
        car_y = self.node.odom.pose.pose.position.y
        car_yaw = euler_from_quaternion(self.node.odom.pose.pose.orientation)
        poses = self.node.path.poses
        
        # 가장 가까운 경로점 찾기
        min_dist_sq = float('inf')
        closest_idx = 0
        
        # 전체 탐색은 느릴 수 있으니 나중엔 최적화 가능 (일단은 안전하게 전체 탐색)
        for i, p in enumerate(poses):
            dx = p.pose.position.x - car_x
            dy = p.pose.position.y - car_y
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = i
        
        closest_pt = poses[closest_idx].pose.position
        
        # 현재 트랙 방향 (Tangent)
        next_idx = (closest_idx + 5) % len(poses) # 5칸 앞을 봐서 노이즈 감소
        track_dx = poses[next_idx].pose.position.x - closest_pt.x
        # [수정 완료] 여기서 closest_ht -> closest_pt 로 수정됨!
        track_dy = poses[next_idx].pose.position.y - closest_pt.y 
        track_yaw = math.atan2(track_dy, track_dx)
        
        # 헤딩 에러
        heading_error = track_yaw - car_yaw
        while heading_error > math.pi: heading_error -= 2 * math.pi
        while heading_error < -math.pi: heading_error += 2 * math.pi
        
        # 횡방향 에러 (Lateral Error)
        dx_vec = car_x - closest_pt.x
        dy_vec = car_y - closest_pt.y
        lateral_error = -math.sin(track_yaw) * dx_vec + math.cos(track_yaw) * dy_vec
        
        # 미래 곡률 (Curvature)
        far_idx = (closest_idx + 20) % len(poses) # 20칸 앞
        far_dx = poses[far_idx].pose.position.x - poses[next_idx].pose.position.x
        far_dy = poses[far_idx].pose.position.y - poses[next_idx].pose.position.y
        far_yaw = math.atan2(far_dy, far_dx)
        
        curvature = far_yaw - track_yaw
        while curvature > math.pi: curvature -= 2 * math.pi
        while curvature < -math.pi: curvature += 2 * math.pi

        return lateral_error, heading_error, curvature

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
        
        self.odom = None
        self.scan = None
        self.collision = False
        self.path = None
        
        self.get_logger().info("토픽 기다리는 중...")
        while rclpy.ok() and (self.path is None or self.odom is None):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("모든 토픽 준비 완료! 학습 시작!")

    def odom_cb(self, msg): self.odom = msg
    def scan_cb(self, msg): self.scan = msg
    def collision_cb(self, msg): self.collision = msg.data
    def path_cb(self, msg): 
        if self.path is None: self.get_logger().info("Path Received!")
        self.path = msg

def main():
    rclpy.init()
    node = RLNode()
    env = F1TenthEnv(node)
    
    print("최종 학습 시작!!! 이제 진짜 달릴 거야!!!")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, device='cpu', batch_size=64, n_steps=2048)
    
    # 100만 스텝 학습
    model.learn(total_timesteps=1000000)
    model.save("f1tenth_advanced")

if __name__ == '__main__':
    main()
