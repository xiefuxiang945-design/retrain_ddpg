import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from Test_env_combine import DDPGEnvironment
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Initialize environment
num_ground_stations = 6
num_uav = 1
state_dim = num_ground_stations * 3 + 3 * num_ground_stations
action_dim = num_ground_stations + 3
width = 200
height = 200
env = DDPGEnvironment(num_ground_stations, width, width, height=height)
# baseline_env=BaselineEnvironment(num_ground_stations,width,width, height=height)
font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')


# 定义一个函数来绘制轨迹，并减少标记的频率
def plot_trajectory(ax, x_data, y_data, z_data, label, color, marker, step=5, markersize=40):
    ax.plot([x * width for x in x_data],
            [y * width for y in y_data],
            [z * height for z in z_data],
            label=label, color=color)
    ax.scatter([x * width for x in x_data[::step]],
               [y * width for y in y_data[::step]],
               [z * height for z in z_data[::step]],
               color=color, marker=marker, s=markersize)


# Actor Network
class Actor(nn.Module):
    def __init__(self, hidden_dim=512, hidden_dim1=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, action_dim)

    def forward(self, x):
        a = torch.relu(self.fc1(x))
        b = torch.relu(self.fc2(a))
        c = torch.tanh(self.fc3(b))  # Pendulum action space is [-1, 1]
        return c


current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + 'ddpg_actor_20240707174820-1.pth'

actor = Actor().to(device)
actor.load_state_dict(torch.load(actor_path))

###############优化方法#################

num_episodes = 100
num_step = 100
action_bound = [-1, 1]
REWARD_BUFFER = np.empty(shape=num_episodes)
Data_rate_BUFFER = np.empty(shape=num_episodes)
Throughput_BUFFER = np.empty(shape=num_episodes)
Energy_efficiency_BUFFER = np.empty(shape=num_episodes)

UAV_trajectory_x = np.zeros((1, num_step))
UAV_trajectory_y = np.zeros((1, num_step))
UAV_trajectory_z = np.zeros((1, num_step))
GT_trajectory_x = np.zeros((1, num_step))
GT_trajectory_y = np.zeros((1, num_step))
GT_trajectory_z = np.zeros((1, num_step))

REWARD_BUFFER_noisac = np.empty(shape=num_episodes)
Data_rate_BUFFER_noisac = np.empty(shape=num_episodes)
Throughput_BUFFER_noisac = np.empty(shape=num_episodes)
Energy_efficiency_BUFFER_noisac = np.empty(shape=num_episodes)

UAV_trajectory_x_noisac = np.zeros((1, num_step))
UAV_trajectory_y_noisac = np.zeros((1, num_step))
UAV_trajectory_z_noisac = np.zeros((1, num_step))
GT_trajectory_x_noisac = np.zeros((1, num_step))
GT_trajectory_y_noisac = np.zeros((1, num_step))
GT_trajectory_z_noisac = np.zeros((1, num_step))

Max_reward = 0
for episode_i in range(num_episodes):
    env.__init__(num_ground_stations, width, width, height=height)
    state = env.reset(0)
    state_noisac = env.reset(0)

    episode_reward = 0
    episode_throughput = 0
    episode_data_rate = 0
    episode_Energy_efficiency = 0

    episode_reward_noisac = 0
    episode_throughput_noisac = 0
    episode_data_rate_noisac = 0
    episode_Energy_efficiency_noisac = 0

    UAV_positions_x = []
    UAV_positions_y = []
    UAV_positions_z = []
    UAV_positions_x.append(state[0])
    UAV_positions_y.append(state[1])
    UAV_positions_z.append(state[2])
    GT_positions_x = []
    GT_positions_y = []
    GT_positions_z = []
    GT_positions_x.append(state[3:3 + num_ground_stations])
    GT_positions_y.append(state[3 + num_ground_stations:3 + 2 * num_ground_stations])
    GT_positions_z.append(state[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])

    UAV_positions_x_noisac = []
    UAV_positions_y_noisac = []
    UAV_positions_z_noisac = []
    UAV_positions_x_noisac.append(state_noisac[0])
    UAV_positions_y_noisac.append(state_noisac[1])
    UAV_positions_z_noisac.append(state_noisac[2])
    GT_positions_x_noisac = []
    GT_positions_y_noisac = []
    GT_positions_z_noisac = []
    GT_positions_x_noisac.append(state_noisac[3:3 + num_ground_stations])
    GT_positions_y_noisac.append(state_noisac[3 + num_ground_stations:3 + 2 * num_ground_stations])
    GT_positions_z_noisac.append(state_noisac[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])

    state_baseline = env.reset(1)
    episode_reward_baseline = 0
    episode_throughput_baseline = 0
    episode_data_rate_baseline = 0
    episode_Energy_efficiency_baseline = 0
    reward_baseline = 0
    UAV_positions_x_baseline = []
    UAV_positions_y_baseline = []
    UAV_positions_z_baseline = []
    UAV_positions_x_baseline.append(state_baseline[0])
    UAV_positions_y_baseline.append(state_baseline[1])
    UAV_positions_z_baseline.append(state_baseline[2])
    GT_positions_x_baseline = []
    GT_positions_y_baseline = []
    GT_positions_z_baseline = []
    GT_positions_x_baseline.append(state_baseline[3:3 + num_ground_stations])
    GT_positions_y_baseline.append(state_baseline[3 + num_ground_stations:3 + 2 * num_ground_stations])
    GT_positions_z_baseline.append(state_baseline[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])
    for step_i in range(num_step):
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        # print(action)
        next_state, reward, done, throughput_step, data_rate_step, Energy_efficiency_step, _ = env.step(action, state,
                                                                                                        0)

        GT_positions_x.append(state[3:3 + num_ground_stations])
        GT_positions_y.append(state[3 + num_ground_stations:3 + 2 * num_ground_stations])
        GT_positions_z.append(state[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])
        UAV_positions_x.append(state[0])
        UAV_positions_y.append(state[1])
        UAV_positions_z.append(state[2])
        state = next_state

        episode_reward += reward
        episode_throughput += throughput_step
        episode_data_rate += data_rate_step
        episode_Energy_efficiency += Energy_efficiency_step

        action_noisac = actor(torch.FloatTensor(state_noisac).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state_noisac, reward_noisac, done, throughput_step_noisac, data_rate_step_noisac, Energy_efficiency_step_noisac, _ = env.step(
            action_noisac, state_noisac, 2)
        GT_positions_x_noisac.append(state_noisac[3:3 + num_ground_stations])
        GT_positions_y_noisac.append(state_noisac[3 + num_ground_stations:3 + 2 * num_ground_stations])
        GT_positions_z_noisac.append(state_noisac[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])
        UAV_positions_x_noisac.append(state_noisac[0])
        UAV_positions_y_noisac.append(state_noisac[1])
        UAV_positions_z_noisac.append(state_noisac[2])
        state_noisac = next_state_noisac

        episode_reward_noisac += reward_noisac
        episode_throughput_noisac += throughput_step_noisac
        episode_data_rate_noisac += data_rate_step_noisac
        episode_Energy_efficiency_noisac += Energy_efficiency_step_noisac

        action_baseline = env.baseline(state_baseline)
        # print(step_i,action_baseline)
        # print(action)
        next_state_baseline, reward_baseline, done, throughput_bs_baseline, data_rate_bs_baseline, Energy_efficiency_bs_baseline, _ = env.step(
            action_baseline, state_baseline, 1)
        GT_positions_x_baseline.append(state_baseline[3:3 + num_ground_stations])
        GT_positions_y_baseline.append(state_baseline[3 + num_ground_stations:3 + 2 * num_ground_stations])
        GT_positions_z_baseline.append(state_baseline[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])
        UAV_positions_x_baseline.append(state_baseline[0])
        UAV_positions_y_baseline.append(state_baseline[1])
        UAV_positions_z_baseline.append(state_baseline[2])
        episode_reward_baseline += reward_baseline
        episode_throughput_baseline += throughput_bs_baseline
        episode_data_rate_baseline += data_rate_bs_baseline
        episode_Energy_efficiency_baseline += Energy_efficiency_bs_baseline
        state_baseline = next_state_baseline

        if done:
            break

    REWARD_BUFFER[episode_i] = episode_reward
    Data_rate_BUFFER[episode_i] = episode_data_rate / num_step
    Throughput_BUFFER[episode_i] = episode_throughput / num_step
    Energy_efficiency_BUFFER[episode_i] = episode_Energy_efficiency / num_step

    REWARD_BUFFER_noisac[episode_i] = episode_reward_noisac
    Data_rate_BUFFER_noisac[episode_i] = episode_data_rate_noisac / num_step
    Throughput_BUFFER_noisac[episode_i] = episode_throughput_noisac / num_step
    Energy_efficiency_BUFFER_noisac[episode_i] = episode_Energy_efficiency_noisac / num_step

    if Max_reward < episode_reward:
        Max_reward = episode_reward

        UAV_trajectory_x = UAV_positions_x
        UAV_trajectory_y = UAV_positions_y
        UAV_trajectory_z = UAV_positions_z

        GT_trajectory_x = GT_positions_x
        GT_trajectory_y = GT_positions_y
        GT_trajectory_z = GT_positions_z

        UAV_trajectory_x_noisac = UAV_positions_x_noisac
        UAV_trajectory_y_noisac = UAV_positions_y_noisac
        UAV_trajectory_z_noisac = UAV_positions_z_noisac

        GT_trajectory_x_noisac = GT_positions_x_noisac
        GT_trajectory_y_noisac = GT_positions_y_noisac
        GT_trajectory_z_noisac = GT_positions_z_noisac

        UAV_trajectory_x_baseline = UAV_positions_x_baseline
        UAV_trajectory_y_baseline = UAV_positions_y_baseline
        UAV_trajectory_z_baseline = UAV_positions_z_baseline

        GT_trajectory_x_baseline = GT_positions_x_baseline
        GT_trajectory_y_baseline = GT_positions_y_baseline
        GT_trajectory_z_baseline = GT_positions_z_baseline
        REWARD_BUFFER_baseline = np.zeros(shape=num_episodes)
        Data_rate_BUFFER_baseline = np.zeros(shape=num_episodes)
        Throughput_BUFFER_baseline = np.zeros(shape=num_episodes)
        Energy_efficiency_BUFFER_baseline = np.zeros(shape=num_episodes)
        for i in range(num_episodes):
            REWARD_BUFFER_baseline[i] = episode_reward_baseline
            Data_rate_BUFFER_baseline[i] = episode_data_rate_baseline / num_step
            Throughput_BUFFER_baseline[i] = episode_throughput_baseline / num_step
            Energy_efficiency_BUFFER_baseline[i] = episode_Energy_efficiency_baseline / num_step
    print(f"TSP-Reward: {round(episode_reward_baseline, 2)}")
    print(f"DDPG-ISAC-Episode: {episode_i + 1}, Reward: {round(episode_reward, 2)}")
    print(f"DDPG-noISAC-Episode: {episode_i + 1}, Reward: {round(episode_reward_noisac, 2)}")

############基线方法#################
####################################### 图像 #############################

# 设置相对路径
relative_path = 'plots'

# 如果目录不存在，则创建目录
if not os.path.exists(relative_path):
    os.makedirs(relative_path)

plt.clf()  # 清除当前图像
# 绘制 REWARD_BUFFER 图
plt.plot(REWARD_BUFFER, marker='o', label='DDPG-ISAC')
plt.plot(REWARD_BUFFER_noisac, label='DDPG')
plt.plot(REWARD_BUFFER_baseline, color='r', linestyle='--', label='Baseline-ISAC')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDPG Reward')
# 指定图例位置
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Reward.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')  # 保存图像，设置分辨率为300 DPI
# plt.show()
plt.close()  # 关闭图像窗口
print('reward', np.mean(REWARD_BUFFER), np.mean(REWARD_BUFFER_noisac), np.mean(REWARD_BUFFER_baseline))

plt.clf()  # 清除当前图像
# 绘制 Data_rate_BUFFER 图
plt.plot(Data_rate_BUFFER, marker='o', label='DDPG-ISAC')
plt.plot(Data_rate_BUFFER_noisac, label='DDPG')
plt.plot(Data_rate_BUFFER_baseline, color='r', linestyle='--', label='Baseline-ISAC')
plt.xlabel('Episode')
plt.ylabel('Data rate (Mbps)')
plt.title('DDPG Data Rate')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Data_rate.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()  # 关闭图像窗口
print('datarate', np.mean(Data_rate_BUFFER), np.mean(Data_rate_BUFFER_noisac), np.mean(Data_rate_BUFFER_baseline))

plt.clf()  # 清除当前图像
# 绘制 Throughput_BUFFER 图
plt.plot(Throughput_BUFFER, marker='o', label='DDPG-ISAC')
plt.plot(Throughput_BUFFER_noisac, label='DDPG')
plt.plot(Throughput_BUFFER_baseline, color='r', linestyle='--', label='Baseline-ISAC')
plt.xlabel('Episode')
plt.ylabel('Throughput (Mb)')
plt.title('DDPG Throughput')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Throughput.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()  # 关闭图像窗口
print('Throughput_BUFFER', np.mean(Throughput_BUFFER), np.mean(Throughput_BUFFER_noisac),
      np.mean(Throughput_BUFFER_baseline))

plt.clf()  # 清除当前图像
# 绘制 Energy_efficiency_BUFFER 图
plt.plot(Energy_efficiency_BUFFER, marker='o', label='DDPG-ISAC')
plt.plot(Energy_efficiency_BUFFER_noisac, label='DDPG')
plt.plot(Energy_efficiency_BUFFER_baseline, color='r', linestyle='--', label='Baseline-ISAC')
plt.xlabel('Episode')
plt.ylabel('Energy Efficiency (Kbps/J)')
plt.title('DDPG Energy Efficiency')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Energy_efficiency.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()  # 关闭图像窗口
print('Energy_efficiency_BUFFER', np.mean(Energy_efficiency_BUFFER), np.mean(Energy_efficiency_BUFFER_noisac),
      np.mean(Energy_efficiency_BUFFER_baseline))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # 添加一个 3D 坐标轴

# 绘制无人机飞行路径
ax.plot([x * width for x in UAV_trajectory_x],
        [y * width for y in UAV_trajectory_y],
        [z * height for z in UAV_trajectory_z],
        label='DDPG-ISAC UAV Flight Path', marker='.', color='blue')

ax.plot([x * width for x in UAV_trajectory_x_noisac],
        [y * width for y in UAV_trajectory_y_noisac],
        [z * height for z in UAV_trajectory_z_noisac],
        label='DDPG UAV Flight Path', marker='.', color='red')

# 绘制基准无人机飞行路径
ax.plot([x * width for x in UAV_trajectory_x_baseline],
        [y * width for y in UAV_trajectory_y_baseline],
        [z * height for z in UAV_trajectory_z_baseline],
        label=' Baseline-ISAC UAV Flight Path', marker='.', color='cyan')

n = num_ground_stations
# 初始化存储位置列表的列表
GT_location_x = [[] for _ in range(n)]
GT_location_y = [[] for _ in range(n)]
GT_location_z = [[] for _ in range(n)]

# 填充存储位置的列表
for slot in GT_trajectory_x:
    for i in range(n):
        GT_location_x[i].append(slot[i])
GT_location_x = [np.array(loc) for loc in GT_location_x]

for slot in GT_trajectory_y:
    for i in range(n):
        GT_location_y[i].append(slot[i])
GT_location_y = [np.array(loc) for loc in GT_location_y]

for slot in GT_trajectory_z:
    for i in range(n):
        GT_location_z[i].append(slot[i])
GT_location_z = [np.array(loc) for loc in GT_location_z]

# 定义一个颜色列表
colors = ['green', 'red', 'purple', 'orange', 'brown', 'pink', 'cyan', 'gray', 'olive']

# 绘图
for i in range(n):
    color = colors[i % len(colors)]  # 循环使用颜色列表
    plot_trajectory(ax, GT_location_x[i], GT_location_y[i], GT_location_z[i], f'GT{i + 1} Path', color, '^', step=5)

# 设置坐标轴标签和图例
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()
ax.set_title('Combined 3D Trajectories of UAV and GT with Baseline')
save_path = os.path.join(relative_path, 'Combined 3D Trajectories of UAV and GT with Baseline.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()

# 生成 2D 图像
fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(111)  # 添加一个 2D 坐标轴

# 绘制无人机飞行路径在XY平面
ax2.plot([x * width for x in UAV_trajectory_x],
         [y * width for y in UAV_trajectory_y],
         label='DDPG-ISAC UAV Flight Path', marker='.', color='blue')

ax2.plot([x * width for x in UAV_trajectory_x_noisac],
         [y * width for y in UAV_trajectory_y_noisac],
         label='DDPG UAV Flight Path', marker='.', color='red')
# 绘制基准无人机飞行路径在XY平面
ax2.plot([x * width for x in UAV_trajectory_x_baseline],
         [y * width for y in UAV_trajectory_y_baseline],
         label='Baseline-ISAC UAV Flight Path', marker='.', color='cyan')

# 绘图
for i in range(n):
    color = colors[i % len(colors)]  # 循环使用颜色列表
    ax2.plot([x * width for x in GT_location_x[i]],
             [y * width for y in GT_location_y[i]],
             label=f'GT{i + 1} Path', marker='^', color=color)

# 设置坐标轴标签和图例
ax2.set_xlabel('X Coordinate')
ax2.set_ylabel('Y Coordinate')
ax2.legend()
ax2.set_title('XY Plane View of Trajectories')
save_path = os.path.join(relative_path, 'XY Plane View of Trajectories.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')