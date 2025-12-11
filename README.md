# Geometric Racing Controller (taemin)

## Algorithm Logic

### 1. Steering Control (Pure Pursuit)
- 차량의 현재 위치에서 `lookahead_distance`만큼 떨어진 경로상의 목표 지점(Goal Point)을 찾음
- 차량의 후륜 중심과 목표 지점 사이의 각도를 계산하여 Ackermann 조향각을 산출
- **수식:** $\delta = \arctan(\frac{L \cdot 2y}{l_d^2})$

### 2. Dynamic Speed Control
- 경로가 휘어지는 정도(곡률, Curvature)를 계산하여 코너에서는 감속하고 직선에서는 가속
- **수식:** $v_{ref} = \max(v_{min}, v_{max} - k_{speed} \cdot |\kappa|)$
- 목표 속도에 도달하기 위해 PID 기반의 가속도 제어를 수행

### 3. Path Planning (Current Status)
- `center_path`, `left_boundary`, `right_boundary`를 모두 구독
- **현재 로직:** 안전을 위해 `center_path`를 기본 주행 경로(`racing_path`)로 설정하여 주행 (추후 최적 경로 생성 로직 확장 가능)

---

## Topics

| Type | Topic Name | Description |
| :--- | :--- | :--- |
| **Sub** | `/odom0` | 차량의 현재 위치 및 속도 (Odometry) |
| **Sub** | `/center_path` | 트랙의 중심 경로 (Global Path) |
| **Sub** | `/left_boundary` | 트랙 좌측 경계 |
| **Sub** | `/right_boundary` | 트랙 우측 경계 |
| **Pub** | `/ackermann_cmd0` | 계산된 조향 및 가속도 명령 |

---

## Parameters (Tunable)

`racing_controller.yaml` 또는 런치 파일에서 파라미터를 수정하여 주행 성능을 튜닝 가능

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `lookahead` | **2.2** | 전방 주시 거리 (m). 작을수록 공격적, 클수록 안정적 |
| `wheelbase` | **0.34** | 차량 축거 (m) |
| `speed_max` | **6.0** | 직선 구간 최대 속도 (m/s) |
| `speed_min` | **2.0** | 코너 구간 최소 속도 (m/s) |
| `k_speed` | **2.5** | 곡률에 따른 감속 계수 (클수록 코너에서 많이 감속) |
| `k_accel` | **1.2** | 가속도 제어 게인 (P-gain) |
| `accel_max` | **10.0** | 최대 가속도 제한 |

---
