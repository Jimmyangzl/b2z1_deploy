
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

ChannelFactoryInitialize(0, 'enp8s0')

# 创建订阅者
sub = ChannelSubscriber('rt/lowstate', LowState_)

# 初始化
sub.Init()

# 读取数据
print('开始接收B2机器人状态数据...')
print('按 Ctrl+C 停止')
print('-' * 50)

try:
    while True:
        # 读取消息
        msg = sub.Read()
        if msg is not None:
            if len(msg.motor_state) > 0:
                print(f'关节0 角度: {msg.motor_state[0].q:.3f}, 速度: {msg.motor_state[0].dq:.3f}')
            print(f'IMU 四元数: {msg.imu_state.quaternion}')
            print(f"foot_force     : {list(msg.foot_force)}")
            print(f"foot_force_est  : {list(msg.foot_force_est)}")
            print('-' * 50)
        time.sleep(0.02)  # 小延时避免CPU过载
except KeyboardInterrupt:
    print('\n程序已停止')
    sub.Close()

