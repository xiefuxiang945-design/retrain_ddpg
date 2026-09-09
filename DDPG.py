import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import os
from Test_env_new import DDPGEnvironment
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# Gate机制类
class GateMechanism(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.f_S = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.f_g = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, vlm_feat, isac_feat):
        g = torch.sigmoid(self.f_S(vlm_feat) + self.f_g(isac_feat))
        fused = g * vlm_feat + (1 - g) * isac_feat
        return fused, g  # 保持返回值与你原有融合函数兼容

def fuse_vlm_isac(vlm_feat, isac_feat, gate_module=None):
    assert vlm_feat.shape == isac_feat.shape, "输入特征维度必须相同"
    assert vlm_feat.shape[0] == 6, "应为6个GT目标"

    feature_dim = vlm_feat.shape[1]
    if gate_module is None:
        gate_module = GateMechanism(feature_dim)
    return gate_module(vlm_feat, isac_feat)


seed_value = 946308
print("Random seed used:", seed_value)
random.seed(seed_value)
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
font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')


# 定义一个函数来绘制轨迹
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
actor_path_isac = model + 'ddpg_actor_20240707174820-1.pth'
actor_isac = Actor().to(device)
actor_isac.load_state_dict(torch.load(actor_path_isac))

actor_path_vlm = model + 'ddpg_actor_20240707205344-11111.pth'
actor_vlm = Actor().to(device)
actor_vlm.load_state_dict(torch.load(actor_path_isac))
###############优化方法#################

num_episodes = 100
num_step = 100
action_bound = [-1, 1]
REWARD_BUFFER_our = np.empty(shape=num_episodes)
Data_rate_BUFFER_our = np.empty(shape=num_episodes)
Throughput_BUFFER_our = np.empty(shape=num_episodes)
Energy_efficiency_BUFFER_our = np.empty(shape=num_episodes)

UAV_trajectory_x_our=np.zeros((1, num_step))
UAV_trajectory_y_our=np.zeros((1, num_step))
UAV_trajectory_z_our=np.zeros((1, num_step))
GT_trajectory_x_our=np.zeros((1, num_step))
GT_trajectory_y_our=np.zeros((1, num_step))
GT_trajectory_z_our=np.zeros((1, num_step))


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
    state_our = env.reset(0)
    # 初始融合 VLM + ISAC 距离特征
    gt_xy = np.stack([
        state_our[3:3 + num_ground_stations],
        state_our[3 + num_ground_stations:3 + 2 * num_ground_stations]
    ], axis=1)
    uav_xy = np.array([state_our[0], state_our[1]])
    gt_xy_real = gt_xy * env.ground_length
    uav_xy_real = uav_xy * env.ground_length

    isac_d = np.linalg.norm(gt_xy_real - uav_xy_real, axis=1)
    vlm_d = isac_d + np.random.normal(0, 5, size=6)

    vlm_tensor = torch.tensor(vlm_d, dtype=torch.float32).view(6, -1)
    isac_tensor = torch.tensor(isac_d, dtype=torch.float32).view(6, -1)

    env.gate_module = GateMechanism(feature_dim=1)  # 初始化 Gate 模块
    fused_tensor, g_weights = fuse_vlm_isac(vlm_tensor, isac_tensor, env.gate_module)
    fused_d = fused_tensor.view(-1).detach().numpy()
    env.vlm_features = [[d] for d in fused_d]
    state = env.reset(0)
    state_noisac = env.reset(0)
    episode_reward_our = 0
    episode_throughput_our = 0
    episode_data_rate_our = 0
    episode_Energy_efficiency_our = 0
    episode_reward_noisac = 0
    episode_throughput_noisac = 0
    episode_data_rate_noisac = 0
    episode_Energy_efficiency_noisac = 0

    state_baseline = env.reset(1)
    episode_reward_baseline = 0
    episode_throughput_baseline = 0
    episode_data_rate_baseline = 0
    episode_Energy_efficiency_baseline = 0

    state_baseline_fusion = env.reset(1)
    episode_reward_baseline_fusion = 0
    episode_throughput_baseline_fusion = 0
    episode_data_rate_baseline_fusion = 0
    episode_Energy_efficiency_baseline_fusion = 0
    # 轨迹初始化
    UAV_positions_x_our = []
    UAV_positions_y_our = []
    UAV_positions_z_our = []
    UAV_positions_x_our.append(state_our[0])
    UAV_positions_y_our.append(state_our[1])
    UAV_positions_z_our.append(state_our[2])
    GT_positions_x_our = []
    GT_positions_y_our = []
    GT_positions_z_our = []
    GT_positions_x_our.append(state_our[3:3 + num_ground_stations])
    GT_positions_y_our.append(state_our[3 + num_ground_stations:3 + 2 * num_ground_stations])
    GT_positions_z_our.append(state_our[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])

    UAV_positions_x_noisac = [state_noisac[0]];
    UAV_positions_y_noisac = [state_noisac[1]];
    UAV_positions_z_noisac = [state_noisac[2]]
    GT_positions_x_noisac = [state_noisac[3:3 + num_ground_stations]]
    GT_positions_y_noisac = [state_noisac[3 + num_ground_stations:3 + 2 * num_ground_stations]]
    GT_positions_z_noisac = [state_noisac[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations]]

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

    reward_baseline_fusion = 0
    UAV_positions_x_baseline_fusion = []
    UAV_positions_y_baseline_fusion = []
    UAV_positions_z_baseline_fusion = []
    UAV_positions_x_baseline_fusion.append(state_baseline_fusion[0])
    UAV_positions_y_baseline_fusion.append(state_baseline_fusion[1])
    UAV_positions_z_baseline_fusion.append(state_baseline_fusion[2])
    GT_positions_x_baseline_fusion = []
    GT_positions_y_baseline_fusion = []
    GT_positions_z_baseline_fusion = []
    GT_positions_x_baseline_fusion.append(state_baseline_fusion[3:3 + num_ground_stations])
    GT_positions_y_baseline_fusion.append(state_baseline_fusion[3 + num_ground_stations:3 + 2 * num_ground_stations])
    GT_positions_z_baseline_fusion.append(state_baseline_fusion[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])
    for step_i in range(num_step):
        # ================================DDPG-VLM-ISAC
        action = actor_isac(torch.FloatTensor(state_our).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state_our, reward, done, throughput_step, data_rate_step, Energy_efficiency_step, fusion_debug = env.step(
            action, state_our, 0)

        print(
            f"Step {step_i} | g: {fusion_debug['g'].flatten().tolist()} | ISAC: {fusion_debug['isac'].flatten().tolist()} | VLM: {fusion_debug['vlm'].flatten().tolist()} | Fused: {fusion_debug['fused'].flatten().tolist()}")
        GT_positions_x_our.append(state_our[3:3 + num_ground_stations])
        GT_positions_y_our.append(state_our[3 + num_ground_stations:3 + 2 * num_ground_stations])
        GT_positions_z_our.append(state_our[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])
        UAV_positions_x_our.append(state_our[0])
        UAV_positions_y_our.append(state_our[1])
        UAV_positions_z_our.append(state_our[2])
        state_our = next_state_our
        episode_reward_our += reward
        episode_throughput_our += throughput_step
        episode_data_rate_our += data_rate_step
        episode_Energy_efficiency_our += Energy_efficiency_step

        # noISAC
        action_noisac = actor_isac(torch.FloatTensor(state_noisac).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state_noisac, reward_noisac, done, throughput_step_noisac, data_rate_step_noisac, Energy_efficiency_step_noisac, _ = env.step(
            action_noisac, state_noisac, 1)
        GT_positions_x_noisac.append(state_noisac[3:3 + num_ground_stations])
        GT_positions_y_noisac.append(state_noisac[3 + num_ground_stations:3 + 2 * num_ground_stations])
        GT_positions_z_noisac.append(state_noisac[3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])
        UAV_positions_x_noisac.append(state_noisac[0]);
        UAV_positions_y_noisac.append(state_noisac[1]);
        UAV_positions_z_noisac.append(state_noisac[2])
        state_noisac = next_state_noisac
        episode_reward_noisac += reward_noisac
        episode_throughput_noisac += throughput_step_noisac
        episode_data_rate_noisac += data_rate_step_noisac
        episode_Energy_efficiency_noisac += Energy_efficiency_step_noisac


        #TSP-ISAC
        action_baseline = env.baseline(state_baseline)
        next_state_baseline, reward_baseline, done, throughput_bs_baseline, data_rate_bs_baseline, Energy_efficiency_bs_baseline, _ = env.step(
            action_baseline, state_baseline, 3)
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
#TSP
        action_baseline_fusion = env.baseline(state_baseline_fusion)
        next_state_baseline_fusion , reward_baseline_fusion , done, throughput_bs_baseline_fusion , data_rate_bs_baseline_fusion , Energy_efficiency_bs_baseline_fusion , _ = env.step(
            action_baseline_fusion , state_baseline_fusion , 2)
        GT_positions_x_baseline_fusion .append(state_baseline_fusion [3:3 + num_ground_stations])
        GT_positions_y_baseline_fusion .append(state_baseline_fusion [3 + num_ground_stations:3 + 2 * num_ground_stations])
        GT_positions_z_baseline_fusion .append(state_baseline_fusion [3 + 2 * num_ground_stations:3 + 3 * num_ground_stations])
        UAV_positions_x_baseline_fusion .append(state_baseline_fusion [0])
        UAV_positions_y_baseline_fusion .append(state_baseline_fusion [1])
        UAV_positions_z_baseline_fusion .append(state_baseline_fusion [2])
        episode_reward_baseline_fusion  += reward_baseline_fusion
        episode_throughput_baseline_fusion  += throughput_bs_baseline_fusion
        episode_data_rate_baseline_fusion  += data_rate_bs_baseline_fusion
        episode_Energy_efficiency_baseline_fusion  += Energy_efficiency_bs_baseline_fusion
        state_baseline_fusion  = next_state_baseline_fusion
        if done:
            break

    # 写入buffer
    REWARD_BUFFER_our[episode_i] = episode_reward_our
    Data_rate_BUFFER_our[episode_i] = episode_data_rate_our / num_step
    Throughput_BUFFER_our[episode_i] = episode_throughput_our / num_step
    Energy_efficiency_BUFFER_our[episode_i] = episode_Energy_efficiency_our / num_step

    REWARD_BUFFER_noisac[episode_i] = episode_reward_noisac
    Data_rate_BUFFER_noisac[episode_i] = episode_data_rate_noisac / num_step
    Throughput_BUFFER_noisac[episode_i] = episode_throughput_noisac / num_step
    Energy_efficiency_BUFFER_noisac[episode_i] = episode_Energy_efficiency_noisac / num_step

    if Max_reward < episode_reward_our:
        Max_reward = episode_reward_our

        UAV_trajectory_x_our = UAV_positions_x_our
        UAV_trajectory_y_our = UAV_positions_y_our
        UAV_trajectory_z_our = UAV_positions_z_our
        GT_trajectory_x_our = GT_positions_x_our
        GT_trajectory_y_our = GT_positions_y_our
        GT_trajectory_z_our = GT_positions_z_our

        UAV_trajectory_x_noisac = UAV_positions_x_noisac;
        UAV_trajectory_y_noisac = UAV_positions_y_noisac;
        UAV_trajectory_z_noisac = UAV_positions_z_noisac
        GT_trajectory_x_noisac = GT_positions_x_noisac;
        GT_trajectory_y_noisac = GT_positions_y_noisac;
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

        UAV_trajectory_x_baseline_fusion = UAV_positions_x_baseline_fusion
        UAV_trajectory_y_baseline_fusion = UAV_positions_y_baseline_fusion
        UAV_trajectory_z_baseline_fusion = UAV_positions_z_baseline_fusion
        GT_trajectory_x_baseline_fusion = GT_positions_x_baseline_fusion
        GT_trajectory_y_baseline_fusion = GT_positions_y_baseline_fusion
        GT_trajectory_z_baseline_fusion = GT_positions_z_baseline_fusion
        REWARD_BUFFER_baseline_fusion = np.zeros(shape=num_episodes)
        Data_rate_BUFFER_baseline_fusion = np.zeros(shape=num_episodes)
        Throughput_BUFFER_baseline_fusion = np.zeros(shape=num_episodes)
        Energy_efficiency_BUFFER_baseline_fusion = np.zeros(shape=num_episodes)
        for i in range(num_episodes):
            REWARD_BUFFER_baseline_fusion[i] = episode_reward_baseline_fusion
            Data_rate_BUFFER_baseline_fusion[i] = episode_data_rate_baseline_fusion / num_step
            Throughput_BUFFER_baseline_fusion[i] = episode_throughput_baseline_fusion / num_step
            Energy_efficiency_BUFFER_baseline_fusion[i] = episode_Energy_efficiency_baseline_fusion / num_step
    print(f"TSP-ISAC-Reward: {round(episode_reward_baseline, 2)}")
    print(f"TSP-VLM-ISAC-Reward: {round(episode_reward_baseline_fusion, 2)}")
    print(f"DDPG-VLM-ISAC-Episode: {episode_i + 1}, Reward: {round(episode_reward_our, 2)}")
    print(f"DDPG-only-Episode: {episode_i + 1}, Reward: {round(episode_reward_noisac, 2)}")


####################################### 图像 #############################

relative_path = 'plots'
if not os.path.exists(relative_path):
    os.makedirs(relative_path)

plt.clf()
plt.plot(REWARD_BUFFER_our, marker='o', label='DDPG-VLM-ISAC(our)')

plt.plot(REWARD_BUFFER_noisac, label='DDPG')
plt.plot(REWARD_BUFFER_baseline_fusion, color='r', linestyle='--', label='TSP-VLM-ISAC')
plt.plot(REWARD_BUFFER_baseline, color='r', linestyle='--', label='TSP-ISAC')

plt.xlabel('Episode');
plt.ylabel('Energy Efficiency (Kbps/J)');
plt.title('DDPG Energy Efficiency')
plt.legend(loc='upper left');
plt.grid()
plt.savefig(os.path.join(relative_path, 'Energy_efficiency.png'), dpi=1200, bbox_inches='tight');
plt.close()
print('reward', np.mean(REWARD_BUFFER_our), np.mean(REWARD_BUFFER_noisac), np.mean(REWARD_BUFFER_baseline_fusion), np.mean(REWARD_BUFFER_baseline))

plt.clf()  # 清除当前图像
# 绘制 Data_rate_BUFFER 图
plt.plot(Data_rate_BUFFER_our, marker='o', label='DDPG-ISAC')
plt.plot(Data_rate_BUFFER_noisac, label='DDPG')
plt.plot(Data_rate_BUFFER_baseline_fusion, color='r', linestyle='--', label='TSP-VLM-ISAC')
plt.plot(Data_rate_BUFFER_baseline, color='r', linestyle='--', label='TSP-ISAC')
plt.xlabel('Episode')
plt.ylabel('Data rate (Mbps)')
plt.title('DDPG Data Rate')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Data_rate.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()  # 关闭图像窗口
print('datarate', np.mean(Data_rate_BUFFER_our), np.mean(Data_rate_BUFFER_noisac), np.mean(Data_rate_BUFFER_baseline_fusion), np.mean(Data_rate_BUFFER_baseline))

plt.clf()  # 清除当前图像
# 绘制 Throughput_BUFFER 图
plt.plot(Throughput_BUFFER_our, marker='o', label='DDPG-ISAC')
plt.plot(Throughput_BUFFER_noisac, label='DDPG')
plt.plot(Throughput_BUFFER_baseline_fusion, color='r', linestyle='--', label='TSP-VLM-ISAC')
plt.plot(Throughput_BUFFER_baseline, color='r', linestyle='--', label='TSP-ISAC')
plt.xlabel('Episode')
plt.ylabel('Throughput (Mb)')
plt.title('DDPG Throughput')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Throughput.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()  # 关闭图像窗口
print('Throughput_BUFFER', np.mean(Throughput_BUFFER_our), np.mean(Throughput_BUFFER_noisac),np.mean(Throughput_BUFFER_baseline_fusion),np.mean(Throughput_BUFFER_baseline))

plt.clf()  # 清除当前图像
# 绘制 Energy_efficiency_BUFFER 图
plt.plot(Energy_efficiency_BUFFER_our, marker='o', label='DDPG-ISAC')
plt.plot(Energy_efficiency_BUFFER_noisac, label='DDPG')
plt.plot(Energy_efficiency_BUFFER_baseline_fusion, color='r', linestyle='--', label='TSP-VLM-ISAC')
plt.plot(Energy_efficiency_BUFFER_baseline, color='r', linestyle='--', label='TSP-ISAC')
plt.xlabel('Episode')
plt.ylabel('Energy Efficiency (Kbps/J)')
plt.title('DDPG Energy Efficiency')
plt.legend(loc='upper left')
plt.grid()
save_path = os.path.join(relative_path, 'Energy_efficiency.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()  # 关闭图像窗口
print('Energy_efficiency_BUFFER', np.mean(Energy_efficiency_BUFFER_our), np.mean(Energy_efficiency_BUFFER_noisac),np.mean(Energy_efficiency_BUFFER_baseline_fusion),np.mean(Energy_efficiency_BUFFER_baseline))



# 3D 轨迹图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot([x * width for x in UAV_trajectory_x_our],
        [y * width for y in UAV_trajectory_y_our],
        [z * height for z in UAV_trajectory_z_our],
        label='DDPG-VLM-ISAC UAV Flight Path', marker='.', color='blue')


ax.plot([x * width for x in UAV_trajectory_x_noisac], [y * width for y in UAV_trajectory_y_noisac],
        [z * height for z in UAV_trajectory_z_noisac], label='DDPG UAV Flight Path', marker='.', color='red')
ax.plot([x * width for x in UAV_trajectory_x_baseline],
        [y * width for y in UAV_trajectory_y_baseline],
        [z * height for z in UAV_trajectory_z_baseline],
        label=' TSP-ISAC UAV Flight Path', marker='.', color='cyan')
ax.plot([x * width for x in UAV_trajectory_x_baseline_fusion],
        [y * width for y in UAV_trajectory_y_baseline_fusion],
        [z * height for z in UAV_trajectory_z_baseline_fusion],
        label=' TSP-VLM-ISAC UAV Flight Path', marker='.', color='orange')
# 初始化存储位置列表的列表
n = num_ground_stations
GT_location_x = [[] for _ in range(n)]
GT_location_y = [[] for _ in range(n)]
GT_location_z = [[] for _ in range(n)]

# 填充存储位置的列表
for slot in GT_trajectory_x_our:
    for i in range(n):
        GT_location_x[i].append(slot[i])
GT_location_x = [np.array(loc) for loc in GT_location_x]

for slot in GT_trajectory_y_our:
    for i in range(n):
        GT_location_y[i].append(slot[i])
GT_location_y = [np.array(loc) for loc in GT_location_y]

for slot in GT_trajectory_z_our:
    for i in range(n):
        GT_location_z[i].append(slot[i])
GT_location_z = [np.array(loc) for loc in GT_location_z]

# 定义一个颜色列表
colors = ['green', 'red', 'purple', 'orange', 'brown','pink', 'cyan', 'gray', 'olive']

# 绘图
for i in range(n):
    color = colors[i % len(colors)]  # 循环使用颜色列表
    plot_trajectory(ax, GT_location_x[i], GT_location_y[i], GT_location_z[i], f'GT{i+1} Path', color, '^', step=5)





ax.set_xlabel('X Coordinate');
ax.set_ylabel('Y Coordinate');
ax.set_zlabel('Z Coordinate')
ax.legend();
ax.set_title('Combined 3D Trajectories of UAV and GT with Baseline')
plt.savefig(os.path.join(relative_path, 'Combined 3D Trajectories.png'), dpi=1200, bbox_inches='tight');
plt.show()

# 2D 轨迹图
fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(111)
ax2.plot([x * width for x in UAV_trajectory_x_our],
        [y * width for y in UAV_trajectory_y_our],
        label='DDPG-VLM-ISAC UAV Flight Path', marker='.', color='blue')
ax2.plot([x * width for x in UAV_trajectory_x_noisac], [y * width for y in UAV_trajectory_y_noisac],
         label='DDPG UAV Flight Path', marker='.', color='red')
ax2.plot([x * width for x in UAV_trajectory_x_baseline],
         [y * width for y in UAV_trajectory_y_baseline],
         label='TSP-ISAC UAV Flight Path', marker='.', color='cyan')
ax2.plot([x * width for x in UAV_trajectory_x_baseline_fusion],
         [y * width for y in UAV_trajectory_y_baseline_fusion],
         label='TSP-VLM-ISAC UAV Flight Path', marker='.', color='orange')

for i in range(n):
    color = colors[i % len(colors)]
    ax2.plot([x * width for x in GT_location_x[i]], [y * width for y in GT_location_y[i]], label=f'GT{i + 1} Path',
             marker='^', color=color)

ax2.set_xlabel('X Coordinate');
ax2.set_ylabel('Y Coordinate')
ax2.legend();
ax2.set_title('XY Plane View of Trajectories')
plt.savefig(os.path.join(relative_path, 'XY Plane View of Trajectories.png'), dpi=1200, bbox_inches='tight');
plt.show()
plt.close()
