#!/usr/bin/env python3
import os
import time
import math
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped

# [중요] 학습할 때 썼던 환경 클래스(F1TenthEnv)를 그대로 가져와야 합니다.
# ppo_rl.py에 있는 F1TenthEnv 클래스와 RLNode 클래스를 여기에 복사하거나 임포트해야 합니다.
# 편의를 위해 아래에 핵심만 다시 적습니다. (ppo_rl.py와 동일해야 함)

def euler_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class F1TenthEnv(gym.Env):
    # ... (ppo_rl.py에 있는 __init__, step, reset, _get_obs 등 모든 내용 복사 붙여넣기) ...
    # ... 코드가 너무 길어지니 ppo_rl.py 내용을 그대로 쓰시되, 아래 main 함수만 바꾸시면 됩니다.
    pass 

# ppo_rl.py 파일을 모듈로 불러와서 쓰는 것이 가장 깔끔합니다.
# 같은 폴더에 있다면 아래와 같이 import 할 수 있습니다.
from ppo_rl import F1TenthEnv, RLNode 

def main():
    rclpy.init()
    node = RLNode()
    
    # 학습 때와 똑같은 환경 구성
    def make_env():
        env = F1TenthEnv(node)
        return env # Monitor는 테스트 때 필수 아님
        
    vecenv = DummyVecEnv([make_env])
    vecenv = VecFrameStack(vecenv, n_stack=4) # 학습 때 4프레임 썼으니 똑같이!
    
    # 모델 불러오기
    model_path = "f1tenth_final" # 또는 가장 최신의 f1tenth_XXXXXX_steps
    
    if os.path.exists(model_path + ".zip"):
        print(f"모델 로드 중... {model_path}")
        model = PPO.load(model_path, env=vecenv)
    else:
        print("모델 파일이 없습니다! 경로를 확인하세요.")
        return

    print("🚗 주행 테스트 시작! (Ctrl+C로 종료)")
    
    obs = vecenv.reset()
    
    try:
        while rclpy.ok():
            # deterministic=True: 학습된 최적의 행동만 하라 (탐험 X)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vecenv.step(action)
            
            # 시각적 확인을 위해 약간의 딜레이를 줄 수도 있음 (선택)
            # time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("테스트 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
