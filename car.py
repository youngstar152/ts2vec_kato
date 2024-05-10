import math
def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''
    waypoints=params['waypoints']
    speed = params['speed']
    closest_waypoints = params['closest_waypoints'] # 車の現在地点から次に最も近い waypoint、前に最も近い waypoint
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    reward=0
  # カーブを曲がるために必要なステアリング角度に基づいて報酬を決定する
  # 次の waypoint への方向を計算

    track_direction = math.atan2(next_point[1] - prev_point[1],next_point[0] - prev_point[0]) # 値はラジアンで返す
    track_direction = math.degrees(track_direction) # 角度で返すように変更
  # 進行方向とトラック方向の差を計算
    heading = params['heading']
    direction_diff = abs(track_direction - heading)

  # カーブと直線では適正速度が異なる
    if speed > 1.8 and not (30 <= direction_diff):
        reward += 4.0 # 直線では加速 
    elif (speed > 1.8) and (30 <= direction_diff):
        reward-=6.0
        
    elif (speed < 2.0) and (30 <= direction_diff):
        reward += 4.0 # 曲線では多少減速
    else:
        reward+=0.5
    
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward += 2.0
    elif distance_from_center <= marker_2:
        reward += 1.5
    
    # 車体の向きと次のwaypointの方向のさが小さい場合に加点
    if direction_diff < 5:
       reward += 6.0
    elif 5 <= direction_diff or  direction_diff < 10:
       reward += 4.0
        
    if params['is_left_of_center']:
       reward+=2.0
    
    if params['is_offtrack']:
       reward-=65
    
    
    return float(reward)
00:33.188


import math
def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    '''
    reward=0
    waypoints=params['waypoints']
    speed = params['speed']
    progress=params['progress']
    if not params['all_wheels_on_track']:
        reward-=2.0
    
    if params['is_offtrack']:
       reward-=60
    
    if progress==0 or progress==20 or progress==40 or progress==60 or progress==80 or progress==100:
       reward+=20
    
    return float(reward)